import os
import shutil
import zipfile
from typing import List
import urllib.request as request


### Download attributes data for selected subject
def download_subject(subject:int, attributes:List[str], root_url:str='https://humbi-dataset.s3.amazonaws.com'):
    # possible attributes = ['body', 'body_texture', 'cloth', 'gaze', 'gaze_texture', 'face', 'face_texture', 'hand']
    output = False
    for attr in attributes:
        part_name = attr + '_subject'
        zip_file = 'subject_%d.zip' % subject
        url = os.path.join(root_url, part_name, zip_file)
        path = '%s_subject_%d.zip' % (attr, subject)

        try:
            print('download %s data for subject %d...' % (attr, subject))
            request.urlretrieve(url, path)
            downloaded_zip = zipfile.ZipFile(path)
            downloaded_zip.extractall()
            os.remove(path)
            output = True # return True
        except request.HTTPError:
            print('%s attribute is missing for subject %d' % (attr, subject))
            return False

    return output


### Get T-pose for selected subject, or first available if can't be found
def get_pose(subject:int, attribute:str):
    pose = '00000025' * (subject <= 453) + '00000131' * (subject > 453) # T-pose

    if attribute in ['body', 'body_texture', 'cloth']:
        attribute_path = 'body'
    elif attribute in ['face', 'face_texture']:
        attribute_path = 'face'
    elif attribute in ['gaze', 'gaze_texture']:
        attribute_path = 'gaze'
    else:
        attribute_path = attribute

    special_subjects = [39, 123, 133, 264, 299, 458, 461, 464, 515, 544, 555, 572, 579, 612, 617]
    special_poses = ['00000209', '00000057', '00000041', '00000121', '00000217', '00000179', '00000211', '00000163', '00000355', '00000179', '00000339', '00000147', '00000115', '00000147', '00000179']
    if subject in special_subjects:
        new_pose = special_poses[special_subjects.index(subject)]
        print('pose %s for subject %d is not T-pose, we replace with a manually selected pose : %s' % (pose, subject, new_pose))
        pose = new_pose

    poses_path = os.path.join('subject_%d' % subject, attribute_path)
    if pose not in os.listdir(poses_path):
        new_pose = sorted(os.listdir(poses_path))[0]
        print('pose %s not available for subject %d, we replace with first pose available : %s' % (pose, subject, new_pose))
        pose = new_pose

    return pose


### Clean disk
def remove_subject(subject:int, target_dir=os.getcwd()):
    subject_directory = os.path.join(target_dir, 'subject_%d' % subject)
    shutil.rmtree(subject_directory)
    return
