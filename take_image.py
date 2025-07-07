#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class TakePhoto:
    def __init__(self):
        rospy.init_node('take_photo_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.got_image = False

    def image_callback(self, msg):
        if not self.got_image:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite('output.jpg', cv_image)
                rospy.loginfo("Saved output.jpg")
                self.got_image = True
                rospy.signal_shutdown("Photo saved, shutting down.")
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {}".format(e))
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    TakePhoto().run()