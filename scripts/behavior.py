#!/usr/bin/env python
import rospy
import tf
import time
import threading
import math
import operator
import random
import numpy as np
import json
import os
import yaml
from dynamic_reconfigure.server import Server
import dynamic_reconfigure.client
from r2_behavior.cfg import BehaviorConfig
from blender_api_msgs.msg import Target, EmotionState, SetGesture
from std_msgs.msg import String, Float64
from r2_perception.msg import Float32XYZ, CandidateFace, CandidateHand, CandidateSaliency
from enum import Enum


class State(Enum):
    INITIALIZE      = 0  # to start the state machine
    IDLE            = 1  # robot is looking around, smiling every now and then, wideangle and realsense turned up, eyes turned down
    INTERESTED      = 2  # robot looks at saliency, blinking extra times
    AMAZED          = 3  # robot looks at saliency or hand, eyes open, eyebrows raised, smiling or mouth slightly open
    USERS_IDLE      = 4  # robot is aware of face(s), looking at the face(s) sometimes, eyecontact sometimes
    USERS_AMAZED    = 5  # robot is aware of face(s), and there is saliency or hand, eyes open, eyebrows raised, smiling or mouth slightly open
    USERS_SPEAKING  = 6  # robot is speaking, looking at the face(s) sometimes, eyecontact sometimes, maybe catch tags from TTS?
    USERS_LISTENING = 7  # robot is listening, looking/staring at the face that speaks (one for now), eyecontact full

    # speaking/listening behavior as per rough video analysis early december 2017


class LookAt(Enum):
    IDLE      = 0  # look at nothing in particular
    AVOID     = 1  # actively avoid looking at face, hand or saliency
    SALIENCY  = 2  # look at saliency and switch
    HAND      = 3  # look at hand
    ONE_FACE  = 4  # look at single face and make eye contact
    ALL_FACES = 5  # look at all faces, make eye contact and switch


class EyeContact(Enum):
    IDLE      = 0  # don't make eye contact
    LEFT_EYE  = 1  # look at left eye
    RIGHT_EYE = 2  # look at right eye
    BOTH_EYES = 3  # switch between both eyes
    TRIANGLE  = 4  # switch between eyes and mouth


class Config:
    def __init__(self):
        self.saliency_count = 10  # timer counts between saliency switch
        self.faces_count = 10  # timer counts between face switch
        self.eyes_count = 10  # timer counts between eye switch
        self.keep_time = 0.5  # time to keep observations around as useful


class Behavior:
    def __init__(self):
        self.robot_name = rospy.get_param("/robot_name")
        self.behavior_srv = Server(BehaviorConfig, self.HandleConfig, namespace='behavior')
        self.lock = threading.Lock()
        self.config = Config()

        # setup face, hand and saliency structures
        self.faces = {}  # index = cface_id, which should be relatively steady from vision_pipeline
            # it contains a dictionary indexed with ts, so faces are maintained similarly to saliency vectors
        self.current_face = 0  # cface_id of current face
        self.last_face = 0  # most recent cface_id of added face
        self.last_face_ts = 0  # most recent ts of added face
        self.hands = {}  # index = ts, it assumes only one hand, old hands removed after time
        self.last_hand_ts = 0  # most recent ts of added hand
        self.saliencies = {}  # index = ts, and old saliency vectors will be removed after time
        self.current_saliency = 0  # ts of current saliency vector
        self.current_eye = 0  # current eye (0 = left, 1 = right, 2 = mouth)

        # counters
        self.saliency_counter = self.config.saliency_count  # counter to switch between saliency
        self.faces_counter = self.config.faces_count  # counter to switch between faces
        self.eyes_counter = self.config.eyes_count  # counter to switch between eyes/mouth for eyecontact

        # current
        # setup state machine
        self.lookat = LookAt.IDLE
        self.eyecontact = EyeContact.IDLE

        self.state = State.INITIALIZE

        # take candidate streams exactly like RealSense Tracker until fusion is better defined and we can rely on combined camera stuff
        rospy.Subscriber('/{}/perception/realsense/cface'.format(self.robot_name), CandidateFace, self.HandleFace)
        rospy.Subscriber('/{}/perception/realsense/chand'.format(self.robot_name), CandidateHand, self.HandleHand)
        rospy.Subscriber('/{}/perception/wideangle/csaliency'.format(self.robot_name), CandidateSaliency, self.HandleSaliency)

        self.head_focus_pub = rospy.Publisher('/blender_api/set_face_target', Target, queue_size=1)
        self.gaze_focus_pub = rospy.Publisher('/blender_api/set_gaze_target', Target, queue_size=1)
        self.expressions_pub = rospy.Publisher('/blender_api/set_emotion_state', EmotionState, queue_size=1)
        self.gestures_pub = rospy.Publisher('/blender_api/set_gesture', SetGesture, queue_size=1)

        self.hand_events_pub = rospy.Publisher('/hand_events', String, queue_size=1)

        self.tf_listener = tf.TransformListener(False, rospy.Duration(1))

        # and start the timer at 10Hz
        self.timer = rospy.Timer(rospy.Duration(0.1),self.HandleTimer)

        # initialize first state
        self.SetState(State.IDLE)


    def SetHeadFocus(self,pos):
        # publish head focus message
        msg = Target()
        msg.x = pos.x
        msg.y = pos.y
        msg.z = pos.z
        msg.speed = 5.0
        self.head_focus_pub.publish(msg)


    def HandleTimer(self,data):

        # this is the heart of the synthesizer, here the lookat and eyecontact state machines take care of where the robot is looking, and random expressions and gestures are triggered to look more alive (like RealSense Tracker)

        ts = data.current_expected

        # ==== handle lookat
        if self.lookat == LookAt.IDLE:
            # no specific target, let Blender do it's soma cycle thing
            ()

        elif self.lookat == LookAt.AVOID:
            # TODO: find out where there is no saliency, hand or face
            # TODO: head_focus_pub
            ()

        elif self.lookat == LookAt.SALIENCY:
            self.saliency_counter -= 1
            if self.saliency_counter == 0:
                self.saliency_counter = self.config.saliency_count
                # TODO: switch to next saliency vector
            # TODO: head_focus_pub to current saliency vector

        elif self.lookat == LookAt.HAND:
            # stare at hand
            if len(self.hands) > 0:
                curhand = self.hands[self.last_hand_ts]
                self.SetHeadFocus(curhand.position)

        else:
            if self.lookat == LookAt.ALL_FACES:
                self.faces_counter -= 1
                if self.faces_counter == 0:
                    self.faces_counter = self.config.faces_count
                    # TODO: switch to next face

            # just take the most recent face for now
            if len(self.faces) > 0:
                curface = self.faces[self.last_face][self.last_face_ts]
                face_pos = Float32XYZ()
                face_pos.x = curface.position.x
                face_pos.y = curface.position.y
                face_pos.z = curface.position.z

                # ==== handle eyecontact (only for LookAt.ONE_FACE and LookAt.ALL_FACES)

                # calculate where left eye, right eye and mouth are on the current face
                left_eye_pos = Float32XYZ()
                right_eye_pos = Float32XYZ()
                mouth_pos = Float32XYZ()

                # all are 5cm in front of the center of the face
                left_eye_pos.x = face_pos.x - 0.05
                right_eye_pos.x = face_pos.x - 0.05
                mouth_pos.x = face_pos.x - 0.05

                left_eye_pos.y = face_pos.y + 0.03  # left eye is 3cm to the left of the center
                right_eye_pos.y = face_pos.y - 0.03  # right eye is 3cm to the right of the center
                mouth_pos.y = face_pos.y  # mouth is dead center

                left_eye_pos.z = face_pos.z + 0.06  # left eye is 6cm above the center
                right_eye_pos.z = face_pos.z + 0.06  # right eye is 6cm above the center
                mouth_pos.z = face_pos.z - 0.04  # mouth is 4cm below the center

                if self.eyecontact == EyeContact.IDLE:
                    # look at center of the head
                    self.SetHeadFocus(face_pos)

                elif self.eyecontact == EyeContact.LEFT_EYE:
                    # look at left eye
                    self.SetHeadFocus(left_eye_pos)

                elif self.eyecontact == EyeContact.RIGHT_EYE:
                    # look at right eye
                    self.SetHeadFocus(right_eye_pos)

                elif self.eyecontact == EyeContact.BOTH_EYES:
                    # switch between eyes back and forth
                    self.eyes_counter -= 1
                    if self.eyes_counter == 0:
                        self.eyes_counter = self.config.eyes_count
                        if self.current_eye == 1:
                            self.current_eye = 0
                        else:
                            self.current_eye = 1
                    # look at that eye
                    if self.current_eye == 0:
                        cur_eye_pos = left_eye_pos
                    else:
                        cur_eye_pos = right_eye_pos
                    self.SetHeadFocus(cur_eye_pos)

                elif self.eyecontact == EyeContact.TRIANGLE:
                    # cycle between eyes and mouth
                    self.eyes_counter -= 1
                    if self.eyes_counter == 0:
                        self.eyes_counter = self.config.eyes_count
                        if self.current_eye == 2:
                            self.current_eye = 0
                        else:
                            self.current_eye += 1
                    # look at that eye
                    if self.current_eye == 0: 
                        cur_eye_pos = left_eye_pos
                    elif self.current_eye == 1:
                        cur_eye_pos = right_eye_pos
                    elif self.current_eye == 2:
                        cur_eye_pos = mouth_pos
                    self.SetHeadFocus(cur_eye_pos)

        # TODO: start random expressions like RealSense Tracker

        # TODO: start random gestures like RealSense Tracker

        prune_before_time = ts - rospy.Duration.from_sec(self.config.keep_time)

        # flush faces dictionary, update current face accordingly, switch away from State.ALL_FACES if one face left, away from State.ONE_FACE if no face left
        for fid in self.faces.keys():
            # list of keys to be removed
            to_be_removed = []
            # find all keys that are earlier than prune_before_time
            for key in self.faces[fid].keys():
                if key < prune_before_time:
                    to_be_removed.append(key)
            # remove the elements
            for key in to_be_removed:
                del self.faces[fid][key]
            # if all elements are removed, remove this face
            if len(self.faces[fid]) == 0:
                del self.faces[fid]
        # TODO: if that means there is only one face left, switch back to State.ONE_FACE
        # TODO: and if it means there is no face left, switch back to State.IDLE (or one of the others)
                
        # flush hands dictionary, switch away from State.AMAZED if no hands left
        to_be_removed = []
        for key in self.hands.keys():
            if key < prune_before_time:
                to_be_removed.append(key)
        # remove the elements
        for key in to_be_removed:
            del self.hands[key]
        # TODO: if all hands are removed, switch back to State.INTERESTED (or one of the others)            

        # flush saliency dictionary, switch away from State.INTERESTED if no saliency vectors left
        to_be_removed = []
        for key in self.saliencies.keys():
            if key < prune_before_time:
                to_be_removed.append(key)
        # remove the elements
        for key in to_be_removed:
            del self.saliencies[key]
        # TODO: if all saliencies are removed, switch back to State.IDLE


    def SetLookAt(self, newlookat):

        if newlookat == self.lookat:
            return

        self.lookat = newlookat

        # initialize new lookat
        if self.lookat == LookAt.IDLE:
            ()

        elif self.lookat == LookAt.AVOID:
            ()

        elif self.lookat == LookAt.SALIENCY:
            # reset saliency switch counter
            self.saliency_counter = self.config.saliency_count

        elif self.lookat == LookAt.HAND:
            ()

        elif self.lookat == LookAt.ONE_FACE:
            # reset eye switch counter
            self.eyes_counter = self.config.eyes_count

        elif self.lookat == LookAt.ALL_FACES:
            # reset eye and face switch counters
            self.faces_counter = self.config.faces_count
            self.eyes_counter = self.config.eyes_count


    def SetEyeContact(self, neweyecontact):

        if neweyecontact == self.eyecontact:
            return

        self.eyecontact = neweyecontact

        # initialize new eyecontact
        if self.eyecontact == EyeContact.IDLE:
            ()

        elif self.eyecontact == EyeContact.LEFT_EYE:
            ()

        elif self.eyecontact == EyeContact.RIGHT_EYE:
            ()

        elif self.eyecontact == EyeContact.BOTH_EYES:
            self.eyes_counter = self.config.eyes_count

        elif self.eyecontact == EyeContact.TRIANGLE:
            self.eyes_counter = self.config.eyes_count


    # ==== MAIN STATE MACHINE

    def SetState(self, newstate):

        # this is where the new main state is initialized, it sets up lookat and eyecontact states appropriately, manage perception system refresh rates and load random gesture and expression probabilities to be processed by HandleTimer

        if newstate == self.state:
            return

        self.state = newstate

        # initialize new state
        if self.state == State.INITIALIZE:
            # this shouldn't happen
            ()

        elif self.state == State.IDLE:
            # TEST: start looking at a face and switch between eyes
            self.SetLookAt(LookAt.ONE_FACE)
            self.SetEyeContact(EyeContact.BOTH_EYES)

        elif self.state == State.INTERESTED:
            ()

        elif self.state == State.AMAZED:
            ()

        elif self.state == State.USERS_IDLE:
            ()

        elif self.state == State.USERS_AMAZED:
            ()

        elif self.state == State.USERS_SPEAKING:
            ()

        elif self.state == State.USERS_LISTENING:
            ()


    def HandleConfig(self, config, level):
        return config


    def HandleFace(self, msg):
        if msg.cface_id not in self.faces.keys():
            self.faces[msg.cface_id] = {}
        self.faces[msg.cface_id][msg.ts] = msg
        self.last_face = msg.cface_id
        self.last_face_ts = msg.ts


    def HandleHand(self, msg):
        self.hands[msg.ts] = msg
        self.last_hand_ts = msg.ts


    def HandleSaliency(self, msg):
        self.saliencies[msg.ts] = msg


if __name__ == "__main__":
    rospy.init_node('behavior')
    node = Behavior()
    rospy.spin()
