import os
import argparse
import json
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import multiprocessing
from itertools import chain
from functools import partial
from llava.visual_prompt_organizer_new import vip_processor, visual_prompt_config, IMG_TOKEN_PLHR


def std_processor(source):
    conv = []
    for i, turn in enumerate(source['conversations']):
        value = turn["value"].replace('<image>\n', IMG_TOKEN_PLHR)
        if turn['from'] == 'gpt':
            new_turn = {"role": "model", "value": value}
        elif turn['from'] == 'human':
            new_turn = {"role": "human", "value": value}
        else:
            raise ValueError(f"Unknown role {turn['from']}")
        conv.append(new_turn)
    return conv


# data_args = type('Args', (), {
#     "data_path": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA-Instruct/vip-llava_stage2_mix.json",
#     "lazy_preprocess": True,
#     "is_multimodal": True,
#     "image_folder": '/fsx-onevision/pengchuanzhang/data/vcr1/datasets01/vcr1/011619',
#     "image_aspect_ratio": 'pad',
#     "image_size_anchor": 336,
#     "vcr_json_path": '/fsx-onevision/pengchuanzhang/data/vcr1/datasets01/vcr1/011619', # {image_folder}/vcr1images should give path to vcr1 json folders
# })()

# input_image_folders = {
#     "coco": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data",
#     "vg": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data",
#     "pointQA_twice": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data", # VG images
#     "v7w": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data/v7w",
#     "flickr30k": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data", # {image_folder}/flickr30k-images should give path to flickr30k image folders
#     "vcr": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data", # {image_folder}/vcr1images should give path to vcr1 image folders
#     "vg_rel": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data",
#     "refcocog": "/data/home/pengchuanzhang/GitHub/ViP-LLaVA/playground/data",
# }


def generate_prompts(process_id, data_args):
    list_data_dict = json.load(open(data_args.data_path, 'r'))
    # get dataset to idx mapping
    task_data_map = defaultdict(list)
    for idx, item in enumerate(list_data_dict):
        if 'image' in item:
            key = item['image'].split('/')[0]
            if type(item['id']) == str and item['id'].split('-')[0] in visual_prompt_config:
                key = item['id'].split('-')[0]
            task_data_map[key].append(idx)
    print("visual_prompt_config.keys(): ", visual_prompt_config.keys())
    print("task_data_map.keys(): ", task_data_map.keys())
    # for COCO, we have grounding data from 315653 to end (364100)
    # for VG, all vg data points (86417) are grounding data
    # Generate the visual prompts
    key = args.dataset
    if key not in task_data_map:
        raise ValueError(f"Specified dataset {key} is not in the data pool {task_data_map.keys()}!")
    idx_list = task_data_map[key]
    print(f"Generating data for {key} at process id {process_id}...")
    conv_data = []
    if key in visual_prompt_config:
        if data_args.num_workers >= 2:
            assert process_id >=0
            chunk_length = len(idx_list) // data_args.num_workers + 1
            idx_list = idx_list[chunk_length*process_id : chunk_length*(process_id+1)]
        if args.max_nums > 0:
            idx_list = idx_list[:args.max_nums]
        for _i, idx in tqdm(enumerate(idx_list)):
            source = list_data_dict[idx]
            image_path = os.path.basename(source['image']) if key in ['v7w'] else source['image']
            image = Image.open(os.path.join(args.image_folder, image_path)).convert('RGB')
            try:
                image, conversation = vip_processor(source, image, image_size_anchor = data_args.image_size_anchor, image_folder = data_args.vcr_json_path, is_test=data_args.is_test)
            except Exception as e:
                print(e)
                print(f"Failed to process {source['id']}, skipping...")
                continue
            # save the image
            short_image_file = os.path.join(key, os.path.splitext(image_path)[0]+f"_{idx}.png")
            output_image_file = os.path.join(args.output_dir, short_image_file)
            output_image_folder = os.path.dirname(output_image_file)
            if not os.path.exists(output_image_folder):
                try:
                    os.makedirs(output_image_folder)
                except Exception as e:
                    print(e)
            image.save(output_image_file)
            conv_data.append({"id": source["id"], "conversation": conversation, "image": short_image_file})
    elif key in ["vg", "coco"]:
        if key == "coco":
            # for COCO, we have grounding data from 315653 to end (364100)
            idx_list = idx_list[315653:]
        for idx in tqdm(idx_list):
            source = list_data_dict[idx]
            conversation = std_processor(source)
            image_path = source['image']
            conv_data.append({"id": source["id"], "conversation": conversation, "image": image_path})
    else:
        raise ValueError(f"Not supported dataset: {key}")
    return conv_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Text processing", add_help=True)
    parser.add_argument(
        "--data_path", type=str, required=True, help="path to vip-llava json file"
    )
    parser.add_argument(
        "--image_folder", type=str, required=True, help="path to the image folder"
    )
    parser.add_argument(
        "--vcr_json_path",
        type=str,
        default='/fsx-onevision/pengchuanzhang/data/vcr1/datasets01/vcr1/011619',
        help="path to the folder that contains vcr json files of metadata",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="output directory", 
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="the key of the dataset to be processed, in ['refcocog', 'vcr', 'vg_rel', 'flickr30k', 'v7w', 'pointQA_twice', 'coco', 'vg'].",
    )
    parser.add_argument(
        "--is_test",
        action="store_true",
        help="whether the dataset is a test dataset",
    )
    parser.add_argument("--image_size_anchor", type=int, default=336)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max_nums", type=int, default=-1)
    args = parser.parse_args()

    if args.num_workers <= 1 or args.dataset not in visual_prompt_config:
        conv_data = generate_prompts(process_id=-1, data_args=args)
    else:
        generate_prompts_with_args = partial(generate_prompts, data_args=args)
        with multiprocessing.Pool() as pool:
            process_ids = range(args.num_workers)
            results = pool.map(generate_prompts_with_args, process_ids)
        conv_data = list(chain.from_iterable(results))
    output_conv_file = os.path.join(args.output_dir, args.dataset+".json")
    with open(output_conv_file, 'w') as f:
        json.dump(conv_data, f, indent=4)