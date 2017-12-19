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
from r2_perception.msg import CandidateFace, CandidateHand, CandidateSaliency
from enum import Enum


class State(Enum):
    SLEEPING        = 0  # robot sleeps, all perception at 1Hz, only very light cycles in Blender, maybe eyes closed, sleep animation?
    IDLE            = 1  # robot is looking around, smiling every now and then, wideangle and realsense turned up, eyes turned down
    INTERESTED      = 2  # robot looks at saliency, blinking extra times
    AMAZED          = 3  # robot looks at saliency or hand, eyes open, eyebrows raised, smiling or mouth slightly open
    USERS_IDLE      = 4  # robot is aware of face(s), looking at the face(s) sometimes, eyecontact sometimes
    USERS_AMAZED    = 5  # robot is aware of face(s), and there is saliency or hand, eyes open, eyebrows raised, smiling or mouth slightly open
    USERS_SPEAKING  = 6  # robot is speaking, looking at the face(s) sometimes, eyecontact sometimes, maybe catch tags from TTS?
    USERS_LISTENING = 7  # robot is listening, looking/staring at the face that speaks (one for now), eyecontact full

    # speaking/listening behavior as per video analysis

class Behavior:
    def __init__(self):
        self.robot_name = rospy.get_param("/robot_name")
        self.behavior_srv = Server(BehaviorConfig, self.HandleConfig, namespace='behavior')
        self.lock = threading.Lock()

        self.state = State.SLEEPING

        # take candidate streams exactly like RealSense Tracker until fusion is better defined
        rospy.Subscriber('/{}/perception/realsense/cface'.format(self.robot_name), CandidateFace, self.HandleFace)
        rospy.Subscriber('/{}/perception/realsense/chand'.format(self.robot_name), CandidateHand, self.HandleHand)
        rospy.Subscriber('/{}/perception/wideangle/csaliency'.format(self.robot_name), CandidateSaliency, self.HandleSaliency)

        self.look_at = rospy.Publisher('/blender_api/set_face_target', Target, queue_size=1)
        self.gaze_at = rospy.Publisher('/blender_api/set_gaze_target', Target, queue_size=1)
        self.expressions_pub = rospy.Publisher('/blender_api/set_emotion_state', EmotionState, queue_size=1)
        self.gestures_pub = rospy.Publisher('/blender_api/set_gesture', SetGesture, queue_size=1)

        self.hand_events_pub = rospy.Publisher('/hand_events', String, queue_size=1)

        self.tf_listener = tf.TransformListener(False, rospy.Duration(1))


    def SetState(self, newstate):

        if newstate == self.state:
            return

        # Things to consider:
        # - random gesture probability pattern
        # - random expression probability pattern
        # - gaze/look targets, soft/hard tracking
        # - blinking
        # - eye contact
        # - perception refresh rate
        self.state = newstate
        if self.state == State.SLEEPING:
            ()
        elif self.state == State.IDLE:
            ()
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
        # make sure the faces are really there, hysteresis: no face -> face is immediate at first sight, face -> no face is after a certain time all faces are gone

        # for now keep one average position for all incoming faces

        # TODO: update average face position        

        if self.state == State.SLEEPING:
            ()
        elif self.state == State.IDLE:
            ()
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


    def HandleHand(self, msg):
        # also hysteresis

        # also keep average position for all incoming hands

        # TODO: update average hand position

        if self.state == State.SLEEPING:
            ()
        elif self.state == State.IDLE:
            ()
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


    def HandleSaliency(self, msg):

        # also keep average position for all incoming hands

        # TODO: update average hand position

        if self.state == State.SLEEPING:
            ()
        elif self.state == State.IDLE:
            ()
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


if __name__ == "__main__":
    rospy.init_node('behavior')
    node = Behavior()
    rospy.spin()
