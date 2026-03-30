
import time
from NatNetClient import NatNetClient
from util import quaternion_to_euler

positions = {}
rotations_rpy = {}
rortations_q = {}


# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame(id, position, rotation_quaternion):
    # Position and rotation received
    positions[id] = position
    # rotations in quaternions
    rortations_q[id] = rotation_quaternion
    # The rotation is in quaternion. We need to convert it to euler angles
    rotx, roty, rotz = quaternion_to_euler(rotation_quaternion)
    # Store the roll pitch and yaw angles
    rotations_rpy[id] = (rotx, roty, rotz)
    



if __name__ == "__main__":
    clientAddress = "192.168.0.37"
    optitrackServerAddress = "192.168.0.4"
    robot_id = 530

    # This will create a new NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_client_address(clientAddress)
    streaming_client.set_server_address(optitrackServerAddress)
    streaming_client.set_use_multicast(True)
    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()

    # while is_running:
    #     print(positions)
    #     time.sleep(1.0)


    while is_running:
        if robot_id in positions:
            # last position
            # print('Last position', positions[robot_id], ' rotation RPY', rotations_rpy[robot_id], ' rotation_q', rortations_q[robot_id])
            print("rotation_q", positions[robot_id])
        #print("rotation_q", rotations_rpy)

        time.sleep(.1)
