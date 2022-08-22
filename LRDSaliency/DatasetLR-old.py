import os, glob, cv2, numpy as np
import random
from random import shuffle

class StaticDataset():
    def __init__(self, frame_basedir, density_basedir, debug, img_size=(480, 288), training_example_props=0.8):
        MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = img_size
        frame_path_list = glob.glob(os.path.join(frame_basedir, '*.*'))
        density_path_list = glob.glob(os.path.join(density_basedir, '*.*'))
        if debug is True:
            print "Debug mode"
            frame_path_list = frame_path_list[:1000]
            density_path_list = density_path_list[:1000]

        self.data = []
        self.labels = []
        print len(frame_path_list)
        for (frame_path, density_path) in zip(frame_path_list, density_path_list):
            # assert frame_path
            frame = cv2.imread(frame_path).astype(np.float32)
            density = cv2.imread(density_path, 0).astype(np.float32)

            frame =self.pre_process_img(frame, False)
            density = self.pre_process_img(density, True)
            self.data.append(frame)
            self.labels.append(density)
            if len(self.data) % 1 == 0:
                print len(self.data), '\r',
        print 'Done'
        self.num_examples = len(self.data)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        self.completed_epoch = 0
        self.index_in_epoch = 0

    def pre_process_img(self, image, greyscale=False):
        if greyscale==False:
            image = image-self.MEAN_VALUE
            image = cv2.resize(image, dsize = self.img_size)
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.
        else:
            image = cv2.resize(image, dsize = self.img_size)
            image = image[None, ...]
            image = image / 255.
        return image

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.completed_epoch += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.data[start:end], self.labels[start:end]

class VideoDataset():
    def __init__(self, frame_basedir,frame_basedir_right, density_basedir, img_size=(112,112), video_length=16, stack=5, bgr_mean_list=[103.939, 116.779, 123.68],sort='bgr'):
        MEAN_VALUE = np.array(bgr_mean_list, dtype=np.float32)   # B G R/ use opensalicon's mean_value
        if sort=='rgb':
            MEAN_VALUE= MEAN_VALUE[::-1]
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = img_size
        self.dataset_dict={}
        self.video_length = video_length
        self.step = 1
        # self.stack = stack
        assert self.step < self.video_length
        self.frame_basedir = frame_basedir
        self.frame_basedir_right = frame_basedir_right
        self.density_basedir = density_basedir
        self.video_dir_list = glob.glob(os.path.join(self.frame_basedir, "*"))
        self.video_dir_list_right = glob.glob(os.path.join(self.frame_basedir_right, "*"))

        # self.setup_video_dataset()

    def setup_video_dataset_c3d(self, overlap=0, training_example_props=0.8, skip_head=11): ## skip those bad data in the previous of a video
        # pass
        self.tuple_list = []
        assert overlap < self.video_length, "overlap should smaller than videolength."
        step = self.video_length - overlap
        #print('step:',step)
        #print('len(self.video_dir_list):',len(self.video_dir_list))
        for i in range(len(self.video_dir_list)):
            video_dir = self.video_dir_list[i]
            #print('video_dir:',video_dir)
            frame_list = glob.glob(os.path.join(video_dir,'*.*'))
            #print('frame_list:',frame_list)
            total_frame = len(frame_list)
            #print('len(frame_list):',len(frame_list))
            
            for j in range(skip_head, total_frame):#, step): ## div 2, so 1/2 of the video_length is overlapped
                if j + self.video_length > total_frame:
                    break
                tup = (i,j) # video index and first frame index
                self.tuple_list.append(tup)
            # print self.tuple_list;exit()
        
        self.num_examples = len(self.tuple_list)

        shuffle(self.tuple_list)
        self.num_training_examples = int(self.num_examples * training_example_props)

        self.training_tuple_list = self.tuple_list[:self.num_training_examples]
        self.validation_tuple_list = self.tuple_list[self.num_training_examples:]
        self.num_validation_examples = len(self.validation_tuple_list)
        print self.num_examples, "samples generated in total,",self.num_training_examples,"training samples,",self.num_validation_examples,"validation samples";#exit()

        self.num_epoch = 0
        self.index_in_training_epoch = 0
        self.index_in_validation_epoch = 0


    def get_frame_LR(self, mini_batch=0, phase='training', density_length='full'):
        ## 
        ## density_length : full, half, one
        if phase == 'training':
            tuple_list = self.training_tuple_list
            index_in_epoch = self.index_in_training_epoch
            self.index_in_training_epoch += mini_batch
            num_examples = self.num_training_examples
            #print(num_examples)
        elif phase == 'validation':
            tuple_list = self.validation_tuple_list
            index_in_epoch = self.index_in_validation_epoch
            self.index_in_validation_epoch += mini_batch
            num_examples = self.num_validation_examples
        
        frame_wildcard = "frame_%d.*"
        gt_wildcard = "frame_%d.*"
        if not index_in_epoch >= num_examples - mini_batch:
            tup_batch = tuple_list[index_in_epoch:index_in_epoch+mini_batch]
        else:
            if phase=='validation':
                self.index_in_validation_epoch = 0
                print "Done for validation"
                return None

            print "One epoch finished, shuffling data..."

            shuffle(self.training_tuple_list)
            self.index_in_training_epoch = 0
            self.num_epoch += 1
            tup_batch = self.training_tuple_list[self.index_in_training_epoch:self.index_in_training_epoch+mini_batch]
            self.index_in_training_epoch += mini_batch

        density_batch = []
        frame_batch = []
        frame_batch_right = []
        for tup in tup_batch:
            #print tup
            current_frame_list = []
            current_frame_list_right = []
            current_density_list = []

            video_index, start_frame_index=tup
            #end_frame_index = start_frame_index + 1#+ self.video_length
   
            video_dir = self.video_dir_list[video_index]
            #video_dir_right = self.video_dir_list_right[video_index]
            video_name = os.path.basename(video_dir)
            #video_name_temp = video_name[0:5]
            #video_name_right = os.path.basename(video_dir_right)
            video_right_path = os.path.join(self.frame_basedir_right, video_name)
            density_dir = os.path.join(self.density_basedir, video_name)
            frame_name = frame_wildcard % start_frame_index
            # print frame_name

            frame_path=  glob.glob(os.path.join(video_dir, frame_name))[0]
            #print('left:',frame_path)
            frame = self.pre_process_img(cv2.imread(frame_path),sort='rgb')
            current_frame_list.append(frame)
          
            frame_path_right=  glob.glob(os.path.join(video_right_path, frame_name))[0]
            #print('right:',frame_path_right)
            #exit()
            frame_right = self.pre_process_img(cv2.imread(frame_path_right),sort='rgb')
            current_frame_list_right.append(frame_right)
            
            density_name = gt_wildcard % start_frame_index
            # print density_dir, density_name
            density_path = glob.glob(os.path.join(density_dir, density_name))[0]
            density = self.pre_process_img(cv2.imread(density_path, 0),True)
            current_density_list.append(density) 
            '''
            if density_length=='full':
                for i in range(start_frame_index,end_frame_index):
                    frame_index = i + 1
                    frame_name = gt_wildcard % frame_index
                    #print frame_name
                    #exit()
                    #print os.path.join(density_dir, frame_name)
                    density_path = glob.glob(os.path.join(density_dir, frame_name))[0]
                    density = self.pre_process_img(cv2.imread(density_path, 0),True)
                    current_density_list.append(density)                
            elif density_length=='one':
                frame_index=end_frame_index
                frame_name = gt_wildcard%frame_index
                density = self.pre_process_img(cv2.imread(glob.glob(os.path.join(density_dir, frame_name))[0], 0), True)
                current_density_list.append(density)
            '''
            frame_batch.append(np.array(frame))
            frame_batch_right.append(np.array(frame_right))
            density_batch.append(np.array(density))
        return frame_batch, frame_batch_right, density_batch

    def pre_process_img(self, image, greyscale=False, sort='rgb'):
        if greyscale==False:
            if sort=='rgb':
                image = image[:, :, ::-1]
            image = image-self.MEAN_VALUE
            #print "image.shape:", self.img_size
            image = cv2.resize(image, dsize = self.img_size)
            #print(image.shape)
            #exit()
            # image = np.transpose(image, (2, 0, 1))
            # image = image[None, ...]
            image = image / 255.
        else:
            image = cv2.resize(image, dsize = self.img_size)
            image = image[...,None]
            image = image / 255.
        return image