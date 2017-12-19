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


class EyeContact:
    def __init__(self):
        self.robot_name = rospy.get_param("/robot_name")
        self.behavior_srv = Server(BehaviorConfig, self.HandleConfig, namespace='behavior')
        self.lock = threading.Lock()


if __name__ == "__main__":
    rospy.init_node('eyecontact')
    node = EyeContact()
    rospy.spin()
