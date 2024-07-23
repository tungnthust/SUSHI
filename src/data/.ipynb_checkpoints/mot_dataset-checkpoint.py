import torch
from src.data.augmentation import GraphAugmentor
import numpy as np
import os.path as osp
from src.data.graph import HierarchicalGraph
from collections import OrderedDict
from torch.nn import functional as F
from src.models.reid.resnet import resnet50_fc256, resnet50_fc512, load_pretrained_weights
from src.models.reid.fastreid_models import load_fastreid_model
from src.data.misc_datasets import BoundingBoxDataset
from torch.utils.data import DataLoader
from src.utils.deterministic import seed_generator, seed_worker

import pandas as pd
import os
import shutil

VIDEO_FPS = 25

class MOTDataset:
    """
    Main dataset class
    """
    def __init__(self, data_info, config, mode):
        assert mode in ('train', 'val', 'test'), "Dataset mode is not valid!"
        self.config = config
        self.df = data_info
        self.mode = mode
        self.seq_names = self.df['seq'].unique().tolist()
        # Index dataset
        self.seq_and_frames = self._index_dataset()
        self.embeddings_dir = self.config.embeddings_dir
        # Sparse index per sequence for val and test datasets
        self.load_or_process_embedding_detections()
        if self.mode in ('val', 'test'):
            self.sparse_frames_per_seq = self._sparse_index_dataset()

    def _index_dataset(self):
        """
        Index the dataset in a form that we can sample
        """
        seq_and_frames = []
        # Loop over the scenes
        for scene in self.seq_names:
            # Get scene specific dataframe
            scene_df = self._get_detections_by_seq_name(scene)
            frames_per_graph = self.config.frames_per_graph

            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list()))
            start_frames = []
            end_frames = []

            # Loop over all frames
            for f in frames:
                if not start_frames or f >= start_frames[-1] + self.config.train_dataset_frame_overlap:
                    valid_frames = np.arange(f, f + frames_per_graph)
                    graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()
                    # Each frame can be a start and end frame only once. To prevent (1, 30), (2, 30) ... (29, 30)
                    if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                            len(graph_df.frame.unique()) >= 2):
                        seq_and_frames.append((scene, graph_df.frame.min(), graph_df.frame.max()))
                        start_frames.append(graph_df.frame.min())
                        end_frames.append(graph_df.frame.max())

        return tuple(seq_and_frames)

    def _sparse_index_dataset(self):
        """
        Overlapping samples used for validation and test. This time we create a dictionary and bookkeep the sequence name
        """
        sparse_frames_per_seq = {}
        frames_per_graph = self.config.frames_per_graph
        overlap_ratio = self.config.evaluation_graph_overlap_ratio

        for scene in self.seq_names:
            scene_df = self._get_detections_by_seq_name(scene)
            sparse_frames = []

            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list()))
            start_frames = []
            end_frames = []

            min_frame = scene_df.frame.min()  # Initializer

            # Continue until all frames are processed
            while len(frames):
                # Valid regions of the df
                valid_frames = np.arange(min_frame, min_frame + frames_per_graph)
                graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()

                # Each frame can be a start and end frame only once. To prevent (1, 30), (2, 30) ... (29, 30)
                if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                        len(graph_df.frame.unique()) >= 2):
                    # Include the sample
                    sparse_frames.append((scene, graph_df.frame.min(), graph_df.frame.max()))

                    # Update start and end frames
                    start_frames.append(graph_df.frame.min())
                    end_frames.append(graph_df.frame.max())

                    # Update the min frame
                    current_frames = sorted(list(graph_df.frame.unique()))
                    num_current_frame = len(current_frames)
                    num_overlaps = round(overlap_ratio * num_current_frame)
                    assert num_overlaps < num_current_frame and num_overlaps > 0, "Evaluation overlap ratio leads to either all frames or no frames"
                    min_frame = current_frames[-num_overlaps]

                    # Remove current frames from the remaining frames list
                    frames = [f for f in frames if f not in current_frames]

                else:
                    current_frames = sorted(list(graph_df.frame.unique()))
                    frames = [f for f in frames if f not in current_frames]
                    min_frame = min(frames)

            # To prevent empty lists
            if sparse_frames:

                # Accumulate sparse_frames_per_seq
                sparse_frames_per_seq[scene] = tuple(sparse_frames)

        return sparse_frames_per_seq

    def _is_dets_and_embeds_ok(self, seq_name):
        # Verify the processed detections file
        node_embeds_path = osp.join(self.embeddings_dir, seq_name, self.config.node_embeddings_dir)
        reid_embeds_path = osp.join(self.embeddings_dir, seq_name, self.config.reid_embeddings_dir)
        try:
            num_frames = len(self._get_detections_by_seq_name(seq_name)['frame'].unique())
            processed_dets_exist = True
        except:
            num_frames = -1
            processed_dets_exist = False

        # Verify the length of the embeddings
        embeds_ok = osp.exists(node_embeds_path) and len(os.listdir(node_embeds_path)) == num_frames
        embeds_ok = embeds_ok and osp.exists(reid_embeds_path) and len(os.listdir(reid_embeds_path)) == num_frames

        # Are both okay?
        return processed_dets_exist and embeds_ok

    def _load_feature_embedding_model(self):
        """
        Load the embedding cnn model to get the embeddings
        """
        transforms = None

        print("REID ARCH??")
        if self.config.reid_arch == 'resnet50_fc512':
            print("RESNET 50 fc512!!")
            feature_embedding_model = resnet50_fc512(num_classes=1000, loss='xent', pretrained=True).to(self.config.device)
            load_pretrained_weights(feature_embedding_model, self.config.feature_embedding_model_path)

        elif self.config.reid_arch.startswith('fastreid_'):
            print("FASTREID MODEL!!")
            feature_embedding_model, transforms =  load_fastreid_model(self.config.reid_arch)

        elif self.config.reid_arch == 'old_model':
            print("OLD MODEL!!")

            #feature_embedding_model = resnet50_fc256(num_classes=2220, loss='xent', pretrained=True).to(self.config.device)
            model_cls = resnet50_fc256 if 'duke' in self.config.feature_embedding_model_path else resnet50_fc512
            num_classes = 2220 if 'duke' in self.config.feature_embedding_model_path else 2968
            feature_embedding_model = model_cls(num_classes=num_classes, loss='xent', pretrained=True).to(self.config.device)
            load_pretrained_weights(feature_embedding_model, self.config.feature_embedding_model_path)
        
        else:
            raise NameError(f"ReID architecture is not {self.config.reid_arch} a valid option")
            
        #load_pretrained_weights(feature_embedding_model, self.config.feature_embedding_model_path)
        return feature_embedding_model, transforms
    
    def _store_embeddings(self, seq_name):
        """
        Stores node and reid embeddings corresponding for each detection in the given sequence.
        Embeddings are stored at:
        Essentially, each set of processed detections (e.g. raw, prepr w. frcnn, prepr w. tracktor) has a storage path, corresponding
        to a detection file (det_file_name). Within this path, different CNNs, have different directories
        (specified in dataset_params['node_embeddings_dir'] and dataset_params['reid_embeddings_dir']), and within each
        directory, we store pytorch tensors corresponding to the embeddings in a given frame, with shape
        (N, EMBEDDING_SIZE), where N is the number of detections in the frame.
        """
        assert self.feature_embedding_model is not None
        assert self.config.reid_embeddings_dir is not None and self.config.node_embeddings_dir

        # Directory paths
        node_embeds_path = osp.join(self.embeddings_dir, seq_name, self.config.node_embeddings_dir)

        reid_embeds_path = osp.join(self.embeddings_dir, seq_name, self.config.reid_embeddings_dir)

        # Delete if exists, and create the directories
        if osp.exists(node_embeds_path):
            print("Found existing stored node embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_embeds_path)
        if osp.exists(reid_embeds_path):
            print("Found existing stored reid embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(reid_embeds_path)
        os.makedirs(node_embeds_path)
        os.makedirs(reid_embeds_path)
        det_df = self._get_detections_by_seq_name(seq_name)

        print(f"Computing embeddings for {det_df.shape[0]} detections")  # Info num detections
        # Make sure that we don't run out of memory, so batch the detections if necessary
        num_dets = det_df.shape[0]
        max_dets_per_df = int(1e5)
        frame_cutpoints = [det_df.frame.iloc[i] for i in np.arange(0, num_dets, max_dets_per_df, dtype=int)]
        frame_cutpoints += [det_df.frame.iloc[-1] + 1]

        # Compute and store embeddings
        for frame_start, frame_end in zip(frame_cutpoints[:-1], frame_cutpoints[1:]):
            # Get the corresponding frames
            sub_df_mask = det_df.frame.between(frame_start, frame_end - 1)
            sub_df = det_df.loc[sub_df_mask]

            # Dataloader
            bbox_dataset = BoundingBoxDataset(sub_df,
                                              return_det_ids_and_frame=True, 
                                              transforms=self.transforms,
                                              output_size=(self.config.reid_img_h, self.config.reid_img_w))
            bbox_loader = DataLoader(bbox_dataset, batch_size=1000, pin_memory=True,
                                     num_workers=self.config.num_workers,
                                     worker_init_fn=seed_worker, generator=seed_generator(),)

            # Feed them to the model
            self.feature_embedding_model.eval()
            node_embeds, reid_embeds = [], []  # Node: before fc layers (2048), reid after fc layers (256)
            frame_nums, det_ids = [], []
            with torch.no_grad():
                for frame_num, det_id, bboxes in bbox_loader:
                    #node_out, reid_out = self.feature_embedding_model(bboxes.to(self.config.device))
                    feature_out = self.feature_embedding_model(bboxes.to(self.config.device))
                    if isinstance(feature_out, torch.Tensor):
                        node_out = feature_out
                        reid_out = feature_out.clone()
                    else:
                        node_out, reid_out = feature_out
                        
                    node_embeds.append(node_out.cpu())
                    reid_embeds.append(reid_out.cpu())
                    frame_nums.append(frame_num)
                    det_ids.append(det_id)

            # Merge with all results
            det_ids = torch.cat(det_ids, dim=0)
            frame_nums = torch.cat(frame_nums, dim=0)
            node_embeds = torch.cat(node_embeds, dim=0)
            reid_embeds = torch.cat(reid_embeds, dim=0)

            # Add detection ids as first column of embeddings, to ensure that embeddings are loaded correctly
            node_embeds = torch.cat((det_ids.view(-1, 1).float(), node_embeds), dim=1)
            reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

            # Save embeddings grouped by frame
            for frame in sub_df.frame.unique():
                mask = frame_nums == frame
                frame_node_embeds = node_embeds[mask]
                frame_reid_embeds = reid_embeds[mask]

                frame_node_embeds_path = osp.join(node_embeds_path, f"{frame}.pt")
                frame_reid_embeds_path = osp.join(reid_embeds_path, f"{frame}.pt")

                torch.save(frame_node_embeds, frame_node_embeds_path)
                torch.save(frame_reid_embeds, frame_reid_embeds_path)

            # print("Finished storing embeddings")
        print("Finished computing and storing embeddings")

    def load_or_process_embedding_detections(self):
        """
        Tries to load a set of processed detections if it's safe to do so. otherwise, it processes them and stores the
        result
        """

        for scene in self.seq_names:
            if self._is_dets_and_embeds_ok(scene):
                print(f"Already processed embedding detections for sequence {scene} ...")
            else:
                print(f'Detections for sequence {scene} need to be processed. Starting processing ...')
                self.feature_embedding_model, self.transforms = self._load_feature_embedding_model()
                self._store_embeddings(scene)


    def _load_precomputed_embeddings(self, det_df, seq_name, embeddings_dir):
        """
        Load the embeddings corresponding to the detections specified in the det_df
        """
        # Retrieve the embeddings we need from their corresponding locations
        embeddings_path = osp.join(self.embeddings_dir, seq_name, embeddings_dir)
        # print("EMBEDDINGS PATH IS ", embeddings_path)
        frames_to_retrieve = sorted(det_df.frame.unique())
        embeddings_list = [torch.load(osp.join(embeddings_path, f"{frame_num}.pt")) for frame_num in frames_to_retrieve]
        embeddings = torch.cat(embeddings_list, dim=0)

        # First column in embeddings is the index. Drop the rows of those that are not present in det_df
        ixs_to_drop = list(set(embeddings[:, 0].int().numpy()) - set(det_df['detection_id']))
        embeddings = embeddings[~np.isin(embeddings[:, 0], ixs_to_drop)]  # Not so clean, but faster than a join
        assert_str = "Problems loading embeddings. Indices between query and stored embeddings do not match. BOTH SHOULD BE SORTED!"
        assert (embeddings[:, 0].numpy() == det_df['detection_id'].values).all(), assert_str

        #embeddings = embeddings[:, 1:]  # Get rid of the detection index (MOVED TO OUT OF THIS FUNCTION)

        return embeddings

    def _get_detections_by_seq_name(self, seq_name):
        return self.df[self.df['seq'] == seq_name]
    
    def get_df_from_seq_and_frames(self, seq_name, start_frame, end_frame):
        """
        Returns a dataframe and a seq_info_dict belonging to the specified sequence range
        """
        # Load the corresponding part of the dataframe
        seq_det_df = self._get_detections_by_seq_name(seq_name)  # Sequence specific dets
        valid_frames = np.arange(start_frame, end_frame + 1)  # Frames to be processed together
        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()  # Take only valid frames
        graph_df = graph_df.sort_values(by=['frame', 'detection_id']).reset_index(drop=True)  # Sort

        return graph_df

    def get_graph_from_seq_and_frames(self, seq_name, start_frame, end_frame):
        """
        Main dataloading function. Returns a hierarchical graph belonging to the specified sequence range
        """
        graph_df = self.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)

        # Ensure that there are at least 2 frames in the sampled graph
        assert len(graph_df['frame'].unique()) > 1, "There aren't enough frames in the sampled graph. Either 0 or 1"

        # Data augmentation
        if self.mode=='train' and self.config.augmentation:
            augmentor = GraphAugmentor(graph_df=graph_df, config=self.config)
            graph_df = augmentor.augment()

        # Load appearance data
        x_reid = self._load_precomputed_embeddings(det_df=graph_df, seq_name=seq_name,
                                                   embeddings_dir=self.config.reid_embeddings_dir)
        
        x_node = self._load_precomputed_embeddings(det_df=graph_df, seq_name=seq_name,
                                                    embeddings_dir=self.config.node_embeddings_dir)


        # Copy node frames and ground truth ids from the dataframe
        x_frame = torch.tensor(graph_df[['detection_id', 'frame']].values)
        x_bbox = torch.tensor(graph_df[['detection_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']].values)
        x_feet = torch.tensor(graph_df[['detection_id', 'feet_x', 'feet_y']].values)
        y_id = torch.tensor(graph_df[['detection_id', 'id']].values)

        # Assert that order of all the loaded values are the same
        assert (x_reid[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_node[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_frame[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_bbox[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_feet[:, 0].numpy() == y_id[:, 0].numpy()).all(), "Feature and id mismatch while loading"

        # Get rid of the detection id index
        x_reid = x_reid[:, 1:]
        x_node = x_node[:, 1:]
        x_frame = x_frame[:, 1:]
        x_bbox = x_bbox[:, 1:]
        x_center = x_bbox[:, :2] + 0.5* x_bbox[:, 2:]
        x_feet = x_feet[:, 1:]
        y_id = y_id[:, 1:]

        if self.config.l2_norm_reid:
            x_reid = F.normalize(x_reid, dim = -1, p=2)
            x_node = F.normalize(x_node, dim = -1, p=2)


        # Further important parameters to pass
        fps = torch.tensor(VIDEO_FPS)
        frames_total = torch.tensor(self.config.frames_per_graph)
        frames_per_level = torch.tensor(self.config.frames_per_level)
        start_frame = torch.tensor(start_frame)
        end_frame = torch.tensor(end_frame)
        # Create the object with float32 and int64 precision and send to the device
        hierarchical_graph = HierarchicalGraph(x_reid=x_reid.float(), x_node=x_node.float(), x_frame=x_frame.long(),
                                               x_bbox=x_bbox.float(), x_feet=x_feet.float(), x_center=x_center.float(), 
                                               y_id=y_id.long(), fps=fps.long(), frames_total=frames_total.long(),
                                               frames_per_level=frames_per_level.long(), 
                                               start_frame=start_frame.long(), end_frame=end_frame.long())

        return hierarchical_graph

    def __len__(self):
        return len(self.seq_and_frames)

    def __getitem__(self, ix):
        seq_name, start_frame, end_frame = self.seq_and_frames[ix]
        return self.get_graph_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)