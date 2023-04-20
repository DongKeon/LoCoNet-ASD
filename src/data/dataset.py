import math
import glob
import random
import torch

from torch.utils import data

import numpy as np
import utils.io as io
import multiprocessing as mp

import pdb


"""ASC-Based"""
class ContextualDataset(data.Dataset):# {{{
    def get_speaker_context(self, ts_to_entity, video_id, target_entity_id,
                            center_ts, candidate_speakers):
        context_entities = list(ts_to_entity[video_id][center_ts])
        random.shuffle(context_entities)
        context_entities.remove(target_entity_id)

        if not context_entities:  
            context_entities.insert(0, target_entity_id)  
            while len(context_entities) < candidate_speakers:
                context_entities.append(random.choice(context_entities))
        elif len(context_entities) < candidate_speakers:
            context_entities.insert(0, target_entity_id) 
            while len(context_entities) < candidate_speakers:
                context_entities.append(random.choice(context_entities[1:]))
        else:
            context_entities.insert(0, target_entity_id)
            context_entities = context_entities[:candidate_speakers]

        return context_entities

    def _decode_feature_data_from_csv(self, feature_data):
        feature_data = feature_data[1:-1]
        feature_data = feature_data.split(',')
        return np.asarray([float(fd) for fd in feature_data])

    def get_time_context(self, entity_data, video_id, target_entity_id,
                         center_ts, half_time_length, stride):
        all_ts = list(entity_data[video_id][target_entity_id].keys())
        center_ts_idx = all_ts.index(str(center_ts))

        start = center_ts_idx-(half_time_length*stride)
        end = center_ts_idx+((half_time_length+1)*stride)
        selected_ts_idx = list(range(start, end, stride))
        selected_ts = []
        for idx in selected_ts_idx:
            if idx < 0:
                idx = 0
            if idx >= len(all_ts):
                idx = len(all_ts)-1
            selected_ts.append(all_ts[idx])

        return selected_ts

    def get_time_indexed_feature(self, video_id, entity_id, selectd_ts):
        time_features = []
        for ts in selectd_ts:
            time_features.append(self.entity_data[video_id][entity_id][ts][0])
        return np.asarray(time_features)

    def _cache_feature_file(self, csv_file):
        entity_data = {}
        feature_list = []
        ts_to_entity = {}

        print('load feature data', csv_file)
        csv_data = io.csv_to_list(csv_file)
        for csv_row in csv_data:
            video_id = csv_row[0]
            ts = csv_row[1]
            entity_id = csv_row[2]
            features = self._decode_feature_data_from_csv(csv_row[-1])
            label = int(float(csv_row[3]))

            # entity_data
            if video_id not in entity_data.keys():
                entity_data[video_id] = {}
            if entity_id not in entity_data[video_id].keys():
                entity_data[video_id][entity_id] = {}
            if ts not in entity_data[video_id][entity_id].keys():
                entity_data[video_id][entity_id][ts] = []
            entity_data[video_id][entity_id][ts] = (features, label)
            feature_list.append((video_id, entity_id, ts))

            # ts_to_entity
            if video_id not in ts_to_entity.keys():
                ts_to_entity[video_id] = {}
            if ts not in ts_to_entity[video_id].keys():
                ts_to_entity[video_id][ts] = []
            ts_to_entity[video_id][ts].append(entity_id)

        print('loaded ', len(feature_list), ' features')
        return entity_data, feature_list, ts_to_entity# }}}


class ASCFeaturesDataset(ContextualDataset):# {{{
    def __init__(self, csv_file_path, time_length, time_stride,
                 candidate_speakers, feat_dim):
        # Space config
        self.time_length = time_length
        self.time_stride = time_stride
        self.candidate_speakers = candidate_speakers
        self.half_time_length = math.floor(self.time_length/2)
        self.feat_dim = feat_dim

        # In memory data
        self.feature_list = []
        self.ts_to_entity = {}
        self.entity_data = {}

        # Load metadata
        self._cache_feature_data(csv_file_path)

    # Parallel load of feature files
    def _cache_feature_data(self, dataset_dir):
        pool = mp.Pool(mp.cpu_count()//2)
        files = glob.glob(dataset_dir)
        results = pool.map(self._cache_feature_file, files)
        pool.close()

        for r_set in results:
            e_data, f_list, ts_ent = r_set
            print('unpack ', len(f_list))
            self.entity_data.update(e_data)
            self.feature_list.extend(f_list)
            self.ts_to_entity.update(ts_ent)
        """
        files = glob.glob(dataset_dir)[:1]
        for f in files:
            e_data, f_list, ts_ent = self._cache_feature_file(f)
            self.entity_data.update(e_data)
            self.feature_list.extend(f_list)
            self.ts_to_entity.update(ts_ent)
        """
            

    def __len__(self):
        return int(len(self.feature_list))

    def __getitem__(self, index):
        video_id, target_entity_id, center_ts = self.feature_list[index]
        entity_context = self.get_speaker_context(self.ts_to_entity, video_id,
                                                  target_entity_id, center_ts,
                                                  self.candidate_speakers)

        target = self.entity_data[video_id][target_entity_id][center_ts][1]
        feature_set = np.zeros((self.candidate_speakers, self.time_length, self.feat_dim*2))
        for idx, ctx_entity in enumerate(entity_context):
            time_context = self.get_time_context(self.entity_data,
                                                 video_id,
                                                 ctx_entity, center_ts,
                                                 self.half_time_length,
                                                 self.time_stride)
            features = self.get_time_indexed_feature(video_id, ctx_entity,
                                                     time_context)
            feature_set[idx, ...] = features

        feature_set = np.asarray(feature_set)
        feature_set = np.swapaxes(feature_set, 0, 2)
        return np.float32(feature_set), target# }}}
