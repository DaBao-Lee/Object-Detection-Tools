from systools import *

def coco_to_txt(annotations_path: str, save_dir: str, use_segments=True) -> None:
    
    from ultralytics.data.converter import convert_coco

    if os.path.exists(save_dir):
        rm_dirs(save_dir)
        
    convert_coco(annotations_path, save_dir=save_dir, use_segments=use_segments, lvis=False, cls91to80=True)
    print('转换完成')

def coco_txt_BaiDu(annotations_path: str, save_dir: str) -> None:
    
    if os.path.exists(save_dir):
        rm_dirs(save_dir)
    create_dirs(save_dir)
    
    for file in os.listdir(annotations_path):
        js = json.load(open(os.path.join(annotations_path, file)))
        images = {f'{x["id"]:d}': x for x in js["images"]}
        
        imgToAnns = defaultdict(list)
        for ann in js["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
        
        for img_id, anns in tqdm(imgToAnns.items()):
            img = images[f"{img_id:d}"]
            f = img["file_name"]
            h, w = img["height"], img["width"]
            bboxes = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:  
                    continue
                cls = ann["category_id"]
                bboxes.append([cls] + box.tolist())
            with open(os.path.join(save_dir, str(f).split('.')[0] + '.txt'), '+a') as t:
                for bbox in bboxes:
                    t.write("%g " % bbox[0])
                    for i in bbox[1:]:
                        t.write("%g " % i)
                    t.write("\n")
    print('转换完成')

def voc_to_txt(annotations_path: str, save_dir: str, obj_dict: dict=None) -> None:
    """
    - obj_dict(dict): 包含对象名称与类别编号的字典 e.g {"wheel": 0, "handle": 1, "base": 2}
    """

    assert save_dir.endswith('/') , 'save_dir 必须以 / 结尾'
    assert obj_dict is not None, '请输入对象名称与类别编号的字典'

    if os.path.exists(save_dir):
        rm_dirs(save_dir)
    create_dirs(save_dir)
    
    xmls = glob(os.path.join(annotations_path, '*.xml'))
    for xml_name in tqdm(xmls):
        txt_name = os.path.basename(xml_name).replace('xml', 'txt')
        f = open(os.path.join(save_dir, txt_name), '+w')
        with open(xml_name, 'rb') as fp:
            xml = etree.HTML(fp.read())
            width = int(xml.xpath('//size/width/text()')[0])
            height = int(xml.xpath('//size/height/text()')[0])
            
            obj = xml.xpath('//object')
            for each in obj:
                name = each.xpath("./name/text()")[0]
                classes = obj_dict[name]
                xmin = int(each.xpath('./bndbox/xmin/text()')[0])
                xmax = int(each.xpath('./bndbox/xmax/text()')[0])
                ymin = int(each.xpath('./bndbox/ymin/text()')[0])
                ymax = int(each.xpath('./bndbox/ymax/text()')[0])
                
                dw = 1 / width
                dh = 1 / height
                x_center = (xmin + xmax) / 2
                y_center = (ymax + ymin) / 2
                w = (xmax - xmin)
                h = (ymax - ymin)
                x, y, w, h = x_center * dw, y_center * dh, w * dw, h * dh
               
                f.write(str(classes) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + '\n')
            f.close()
    print('转换完成')

def rm_icc_profile(src_path: str) -> None:

    src_path = glob(os.path.join(src_path, '*.png'))
    for path in tqdm(src_path):
        img = Image.open(path)
        img.save(path, format="PNG", icc_profile=None)
    
    print('icc_profile 已删除')

def rename_files(src_path: str, des_path: str ="./New_Annotation") -> None:

    if os.path.exists(des_path):
        rm_dirs(des_path)
    create_dirs(des_path)

    files = fetch_specific_files(src_path)
    affix = [os.path.splitext(os.path.basename(x))[-1] for x in files]
    if (n:= 0) or (".txt" in set(affix) or ".xml" in set(affix)):
        label_files, typing = (fetch_specific_files(src_path, "txt"), '.txt') if ".txt" in set(affix) else (fetch_specific_files(src_path, "xml"), '.xml')
        img_files = list(set(files).difference(set(label_files)))
        for img_file in tqdm(img_files):
            original_name, _ = os.path.splitext(os.path.basename(img_file))
            shutil.copy2(img_file, os.path.join(des_path, f"{n}{_}"))
            shutil.copy2(os.path.join(src_path, original_name + typing), os.path.join(des_path, f"{n}{typing}"))
            n += 1
    else:
        assert ".txt" in set(affix) or ".xml" in set(affix), "No label files found."

    print('Rename files success...')

def normalize_labels(src_path: str) -> None:

    src_path = glob(os.path.join(src_path, '*.txt'))
    for path in tqdm(src_path):
        f = open(path, "r")
        texts = f.readlines()
        f.close()
        with open(path, "+w") as t:
            for text in texts:
                text = text.split()
                for tt in text[1:]:
                    if float(tt) >= 1.0:
                        text[text.index(tt)] = '0.99999'
                t.write(text[0] + " ")
                for tt in text[1:]:
                    t.write("%g " % float(tt))
                t.write("\n")
    
    print('Label files normalized over...')

def create_type_dict(path: str) -> dict:

    r = open(path, "r").readlines()
    dic = {k.strip(): v for v, k in enumerate(r)}

    inverse_dict = {v: k for k, v in dic.items()}

    return dic, inverse_dict

def get_type_count(path: Union[str, list], type_dict: dict, verbose: bool=True, need_return: bool=False) -> Union[dict, None]:

    type_count = {}
    if isinstance(path, str):
        assert os.path.exists(path), "Path does not exist."
        txts = fetch_specific_files(path, file_type="txt")
    else: txts = path

    for txt in tqdm(txts):
        r = open(txt, "r").readlines()
        labels = [int(x.split()[0]) for x in r]
        for label in labels:
            type_count[type_dict[label]] = type_count.get(type_dict[label], 0) + 1

    if verbose:
        print("\nType Count:")
        print("-" * 20)
        for key, value in type_count.items():
            print(f"{key:10} {value:5}")
        print("-" * 20)
    
    if need_return: return type_count

def check_and_copy_missing_files(img_path: str, label_path: str = None, label_type: str="txt",
                                  meta_path: str = None) -> None:
    
    imgs_name = [os.path.splitext(os.path.basename(x))[0] for x in glob(os.path.join(img_path, "*.*")) if x.endswith((".jpg", ".png", "jpeg"))]

    if label_path is not None:
        labels_name = [os.path.splitext(os.path.basename(x))[0] for x in glob(os.path.join(label_path, f"*.{label_type}"))]
        if len(imgs_name) >= len(labels_name):
            for name in tqdm(imgs_name):
                if name not in labels_name:
                    print(f"{name} not in labels")
        else:
            for name in tqdm(labels_name):
                if name not in imgs_name:
                    print(f"{name} not in imgs")
    else:
        print("labels_name is None")
    if meta_path is not None:

        meta_name = [os.path.splitext(os.path.basename(x))[0] for x in glob(os.path.join(meta_path, "*.*")) if x.endswith((".jpg", ".png", "jpeg"))]
        for name in tqdm(meta_name):
            if name not in imgs_name:
                print(f"{name} not in imgs")
                
    print('检查完成...')

def split_meta(meta_path: str, split_name: str="meta", split_num: int=5, match: bool=True, label_type: str="txt") -> None:
    
    imgs = []
    for ext in ["jpg", "png", "jpeg"]:
        imgs.extend(glob(meta_path.rstrip('/') + f"/*.{ext}"))
    labels = glob(meta_path.rstrip('/') + f"/*.{label_type}")
    
    if match:
        img_dict = {os.path.splitext(os.path.basename(img))[0]: img for img in imgs}
        label_dict = {os.path.splitext(os.path.basename(label))[0]: label for label in labels}
        common_keys = set(img_dict.keys()).intersection(set(label_dict.keys()))
        imgs = [img_dict[key] for key in common_keys]
        labels = [label_dict[key] for key in common_keys]
    
    if len(imgs) != len(labels): 
        print("imgs and labels not match")
        return 

    length = len(imgs) // split_num + 1
    if os.path.exists('./meta_split_results'):
        shutil.rmtree('./meta_split_results')
    os.mkdir('./meta_split_results')
    for i in range(split_num):
        os.mkdir(f'./meta_split_results/{split_name}{i}')
        for index in range(i*length, length * (i + 1)):
            try:
                shutil.copy2(imgs[index], f'./meta_split_results/{split_name}{i}')
                shutil.copy2(labels[index], f'./meta_split_results/{split_name}{i}')
            except: pass

    print("split done...")

def collcet_meta(meta_split_path: str="./meta_split_results", save: bool=False) -> list:
    
    if not os.path.exists(meta_split_path):
        print("no split results found")
        return

    split_results = glob(f"{meta_split_path.rstrip('/')}/*")
    meta = []
    for split_result in split_results:
        meta.extend(glob(split_result.rstrip('/') + "/*.*"))
    
    print("collect done...")

    if save:
        os.mkdir('./meta')
        process_files(meta, './meta')
        print("meta files saved at ./meta")

    return meta

def train_test_split(img_label_path: Union[str, list], create_dir: bool=False, random_seed: int=42,
        split_ratio: float=0.8, need_test: bool=False, need_negative: Union[bool, str]=False, upset: bool=False):
    
    random.seed(random_seed)
    if upset: rm_dirs('./data')
    folder_name = ["train", "val", "test"]
    if create_dir:
        if not need_test: folder_name = ["train", "val"]
        for folder in folder_name:
            create_dirs("./data/" + folder + "/images")
            create_dirs("./data/" + folder + "/labels")
    
    if isinstance(img_label_path, str):
        img_label_path = fetch_specific_files(img_label_path)
    
    labels = fetch_specific_files(img_label_path, "txt")
    imgs = list(set(img_label_path).difference(set(labels)))
    train_imgs = random.sample(imgs, int(len(imgs) * split_ratio))
    train_labels = [re.sub(r'\.(jpg|png|jpeg)$', '.txt', img) for img in train_imgs]

    val_imgs = list(set(imgs).difference(set(train_imgs)))
    val_labels = [re.sub(r'\.(jpg|png|jpeg)$', '.txt', img) for img in val_imgs]

    if need_test:
        test_imgs = random.sample(train_imgs, int(len(train_imgs) * 0.1))
        test_labels = [re.sub(r'\.(jpg|png|jpeg)$', '.txt', img) for img in test_imgs]

        train_imgs = list(set(train_imgs).difference(set(test_imgs)))
        train_labels = list(set(train_labels).difference(set(test_labels)))
    
    if need_negative != False:
        assert isinstance(need_negative, str), "need_negative must be a string"
        assert os.path.exists(need_negative), "Path does not exist."
        negative_imgs = random.sample(fetch_specific_files(need_negative, 'jpg'), int(len(train_imgs) * 0.1))
        train_imgs += negative_imgs

    for folder in folder_name:
        print(f"Copying files to {folder}/images...")
        process_files(eval(f'{folder}_imgs'), f'./data/{folder}/images/')
        print(f"Copying files to {folder}/labels...")
        process_files(eval(f'{folder}_labels'), f'./data/{folder}/labels/')

def create_yaml(names: Union[list, dict], need_test: bool=False, need_labels: bool=True):

    if isinstance(names, dict):
        names = list(names.values()) if 0 in names.keys() else list(names.keys())

    names = [x.title() for x in names]
    with open("./data/data.yaml", "+w") as f:
        config = {
            "path": os.path.abspath('.') + "/data",
            "train": "./train/images",
            "val": "./val/images",
            "nc": len(names),
            "names": names 
        }
        if need_test:
            config["test"] = "./data/test/images (Optional)"

        yaml.dump(config, f, indent=2)

    if need_labels:
        with open('./classes.txt', 'w') as f:
            for name in names:
                f.write(name + '\n')

def train(model_selection: Union[str, list], yaml_data: str, yolo_world: bool=False, rtdter: bool=False, epochs: int = 100, batch: int = -1, val: bool = True,
           save_period: int = -1, project: str = None, pretrained: str = None, single_cls: bool = False,close_mosaic: int = 10,
             lr0: float = 0.01, lrf: float=0.01, workers: int = 0, seed_change: bool = False, cls: float = 0.5, imgsz: int = 640,
             optimizer="SGD", patience=100, resume: bool = False, plots=True, cos_lr=True, iou:float=0.7, task: str="detect"):
    
    print("Loading Training Model...")
    from ultralytics import YOLO, YOLOWorld, RTDETR

    seed = random.randint(1, 1e9) if seed_change else 1
    if yolo_world: model = YOLOWorld(model_selection)
    elif rtdter: model = RTDETR(model_selection)
    else: 
        if not isinstance(model_selection, list):
            model = YOLO(model_selection)
            print(colorama.Fore.GREEN + "Loading the model structure from the pretrained model...")
        else:
            model = YOLO(model_selection[0]) if model_selection[0].endswith("yaml") else YOLO(model_selection[1])
            print(colorama.Fore.YELLOW + "Loading the model structure from the yaml file...")
            model.load(model_selection[1])

    print(colorama.Fore.WHITE + "Loading Model Success...")
    model.train(data=yaml_data, epochs=epochs, workers=workers, batch=batch,
                save_period=save_period, val=val, pretrained=pretrained,
                project=project, lr0=lr0, lrf=lrf, single_cls=single_cls, imgsz=imgsz,
                  seed=seed, cls=cls, optimizer=optimizer, patience=patience,
                  resume=resume, plots=plots, cos_lr=cos_lr, iou=iou, task=task, close_mosaic=close_mosaic)
    
    print(f'Echo: In this train, we use the seed of {seed}.')

def get_best_last_model(model_path: str = "./runs/detect/train/weights", index: int = None, mode: int = 0) -> str:

    if index is not None:
        model_path = f"./runs/detect/train{index}/weights"

    models = glob(os.path.join(model_path, '*.pt'))
    assert models, f"{model_path} not finds model... "

    if mode == 0:
        path = next(x for x in models if "best" in os.path.basename(x))
        print(f"Fetch the best model: {path}")

    elif mode == 1:
        path = next(x for x in models if "last" in os.path.basename(x))
        print(f"Fetch the last model: {path}")

    print("Model Fetched Success...")

    return str(path)

def evaluate(model_selection: str, yaml_data: str, yolo_world=False, iou:float=0.7, verbose: bool=True, save: bool=False) -> any:
    from ultralytics import YOLO, YOLOWorld
    
    if yolo_world: result = YOLOWorld(model_selection).val(data=yaml_data, iou=iou, save=save)
    else: result = YOLO(model_selection).val(data=yaml_data, iou=iou, save=save)
    
    print("Evaluation finished.")

    if verbose: return result

class InferDataType(Enum):

    IMAGE = "image"
    DIR = "dir"

def get_infer_data(path_dir: str, typing: InferDataType = InferDataType.IMAGE, max_num: int = 5):

    imgs = [x for x in glob(os.path.join(path_dir, '*.*')) if "jpg" in x or "png" in x or "jpeg" in x]

    if typing == InferDataType.IMAGE:
        return random.sample(list(imgs), min(max_num, len(imgs)))

    elif typing == InferDataType.DIR:
        return imgs


def predict(model_selection: str, img_path: str, yolo_world=False, conf: float = 0.5, save: bool = False,
                show: bool = True, verbose: bool = True, stream: bool = False, iou: float = 0.7):
    
    from ultralytics import YOLO, YOLOWorld, RTDETR

    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)
    model(source=img_path, conf=conf, show=show, save=save, verbose=verbose, stream=stream, iou=iou)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_pillow_text(img_array:np.array, font_path: str, text: str, position, textColor=(255, 255, 255), textSize: int = 20):

    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(
        font_path, textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def advanced_predict(model_selection: str, img_path: Union[str, list], yolo_world=False, conf: float = 0.5,
                    save: bool = False, show: bool = True, font_path=None, replace_text: dict = None, iou: float = 0.7):

    from ultralytics import YOLO, YOLOWorld

    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)

    if isinstance(img_path, str) and not img_path.endswith(('.jpg', '.png', '.jpeg')):
        img_path = [file for x in ['jpg', 'png', 'jpeg'] for file in glob(os.path.join(img_path, f'*.{x}'))]
        
    results = model(img_path, conf=conf, show=False, save=False, verbose=False, iou=iou)
    for result in results:
        names = result.names
        if replace_text is not None:
            names = replace_text
        boxes = result.boxes
        orig_img = result.orig_img
        path = result.path

        for index in range(len(boxes)):
            cls_ = names[boxes[index].cls.cpu().item()]
            x1, y1, x2, y2 = np.int_(boxes.xyxy[index].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (220, 110, 10), 3)
            text_x = max(0, x1)
            text_y = y1
            
            text_bg_color = (220, 110, 10)
            cv2.rectangle(orig_img, (text_x, text_y - 1), (text_x + 50, text_y + 23), text_bg_color, -1)
            if font_path is not None:
                orig_img = add_pillow_text(orig_img, font_path, cls_, (text_x, text_y), textSize=22)
            else: cv2.putText(orig_img, cls_, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if show:
            cv2.imshow('result', orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save:
            if not os.path.exists('results'):
                os.mkdir('./results')
            cv2.imwrite("results/" + path.split('/')[-1], orig_img)

    print("Prediction finished.")

def export_model(model_selection: str, yolo_world: bool=False, format: str="onnx"):
    
    from ultralytics import YOLO, YOLOWorld

    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)

    model.export(format=format)

    print("Model exported successfully.")

def show_pic(img: Union[cv2.Mat, str], title: str="Image") -> None:

    if isinstance(img, str):
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
    
    assert img is not None, "Image not found or invalid image path."

    cv2.imshow(title, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # train_test_split(img_label_path='./Annotations/meta/', create_dir=True,
    #                   random_seed=100, upset=True, need_test=False,
    #                   need_negative=False)
    
    # create_yaml(names={'Wheel': 0, 'Handle': 1, 'Base': 2, 'Basket': 3,
    #                    'Pedal': 4, 'Rack': 5, 'Lock': 6, 'Helmet': 7,
    #                      'Bell': 8}, need_test=True)

    train(model_selection=['./yolo11n_Ghost_SPPELAN.yaml', './best.pt'], yaml_data='./data/data.yaml', workers=4, patience=0, 
        epochs=2, batch=24, val=True, lr0=0.01, lrf=0.001, seed_change=True, iou=0.7, optimizer="Adam",
        imgsz=416, single_cls=False, resume=False, close_mosaic=0)

# './yolo11n_Ghost_SPPELAN.yaml'
# yolo11_DyT