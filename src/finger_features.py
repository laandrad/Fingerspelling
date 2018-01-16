import os, sys, inspect
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
lib_dir = os.path.abspath(os.path.join(src_dir, '../lib'))
sys.path.insert(0, lib_dir)

import Leap


class GetFeatures:
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def get_features(self, frame):

        features = {}

        # Get hands
        for hand in frame.hands:

            if hand.is_right:

                hand_x_basis = hand.basis.x_basis
                hand_y_basis = hand.basis.y_basis
                hand_z_basis = hand.basis.z_basis
                hand_origin = hand.palm_position
                hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
                hand_transform = hand_transform.rigid_inverse()

                features['frame.id'] = frame.id

                # Get fingers
                for finger in hand.fingers:

                    transformed_position = hand_transform.transform_point(finger.tip_position)
                    transformed_direction = hand_transform.transform_direction(finger.direction)

                    features['{}_{}'.format(self.finger_names[finger.type],
                                            "position_x")] = transformed_position.x
                    features['{}_{}'.format(self.finger_names[finger.type],
                                            "position_y")] = transformed_position.y
                    features['{}_{}'.format(self.finger_names[finger.type],
                                            "position_z")] = transformed_position.z
                    features['{}_{}'.format(self.finger_names[finger.type],
                                            "direction_x")] = transformed_direction.z
                    features['{}_{}'.format(self.finger_names[finger.type],
                                            "direction_y")] = transformed_direction.z
                    features['{}_{}'.format(self.finger_names[finger.type],
                                            "direction_z")] = transformed_direction.z

                    # Get bones
                    for b in range(0, 4):
                        bone = finger.bone(b)

                        features['{}_{}_{}'.format(self.finger_names[finger.type],
                                                self.bone_names[bone.type],
                                                "direction_x")] = bone.direction.x
                        features['{}_{}_{}'.format(self.finger_names[finger.type],
                                                self.bone_names[bone.type],
                                                "direction_y")] = bone.direction.y
                        features['{}_{}_{}'.format(self.finger_names[finger.type],
                                                self.bone_names[bone.type],
                                                "direction_z")] = bone.direction.z
                        features['{}_{}_{}'.format(self.finger_names[finger.type],
                                                self.bone_names[bone.type],
                                                "center_x")] = bone.center.x
                        features['{}_{}_{}'.format(self.finger_names[finger.type],
                                                self.bone_names[bone.type],
                                                "center_y")] = bone.center.y
                        features['{}_{}_{}'.format(self.finger_names[finger.type],
                                                self.bone_names[bone.type],
                                                "center_z")] = bone.center.z

        return features
