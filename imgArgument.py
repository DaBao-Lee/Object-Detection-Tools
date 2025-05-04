from imgtools import *
import albumentations as A

class ImageArgumentor:

    def __init__(self, enhance_time: Union[int, str]=0, img_path:Union[str, list]=None,
                 argument_output: str=None):
        
        self.enhance_time = enhance_time
        if isinstance(img_path, str) and img_path.endswith(("jpg", "png", "jpeg")): 
            self.img_path = [img_path]
        else:
            self.img_path = fetch_specific_files(img_path)

        if argument_output is not None:
            if not os.path.exists(argument_output):
                os.mkdir(argument_output)
        self.argument_output = argument_output

    def toGray(self, prob: float=0.25):

        transform = A.ToGray(p=prob)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("toGray done")
    
    def addGaussNoise(self, prob: float=0.25, mean: float=0.25, var: float=15):
        transform = A.GaussNoise(p=prob, mean=mean, var=var)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_Guassinaoise done")

    def addGaussianBlur(self, prob: float=0.25, blur_limit: int=3):
        transform = A.GaussianBlur(p=prob, blur_limit=blur_limit)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_GaussianBlur done")

    def addGammaNoise(self, prob: float=0.25, gamma_limit: float=80):
        transform = A.RandomGamma(p=prob, gamma_limit=gamma_limit)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_GammaNoise done")
    
    def addRain(self, prob: float=0.25, slant_lower: int=-10, slant_upper: int=10,
                drop_length: int=20, drop_width: int=1, blur_value: int=3):
        transform = A.RandomRain(p=prob, slant_lower=slant_lower, slant_upper=slant_upper,
                           drop_length=drop_length, drop_width=drop_width, blur_value=blur_value)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_Rain done")

    def addFog(self, prob: float=0.25, fog_coef_lower: float=0.1, fog_coef_upper: float=0.3,
              alpha_coef: float=0.5):
        transform = A.RandomFog(p=prob, fog_coef_lower=fog_coef_lower, fog_coef_upper=fog_coef_upper,
                           alpha_coef=alpha_coef)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_Fog done")

    def addSunGlare(self, prob: float=0.5, angle_lower: float=0, angle_upper: float=1,
                    src_radius: float=100, src_color: tuple=(111, 111, 111)):
        
        transform = A.RandomSunFlare(p=prob, angle_lower=angle_lower,
                                    angle_upper=angle_upper,
                                    src_radius=src_radius,
                                    src_color=src_color)
        
        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)
        
        print("add_SunGlare done")

    def addToneCurve(self, scale: float=0.5, prob: float=0.5) -> None:

        transform = A.RandomToneCurve(p=prob, scale=scale, per_channel=True,
                                      always_apply=True)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_ToneCurve done")

    def addSnow(self, prob: float=0.5, snow_point_lower: int=0.1,
                snow_point_upper: int=0.9, brightness_coeff: float=0.5) -> None:

        transform = A.RandomSnow(p=prob, snow_point_lower=snow_point_lower,
                                 snow_point_upper=snow_point_upper,
                                 brightness_coeff=brightness_coeff)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_Snow done")
        
    def InvertImg(self, prob: float=0.5) -> None:

        transform = A.InvertImg(p=prob)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("InvertImg done")

    def addSharpen(self, prob: float=0.5) -> None:

        transform = A.Sharpen(p=prob)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)
        
        print("add_Sharpen done")

    def addToSepia(self, prob: float=0.5) -> None:

        transform = A.ToSepia(p=prob)

        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("add_ToSepia done")

    def addCompose(self) -> None:

        transform = A.Compose([
            A.GaussNoise(p=1, var_limit=(10, 12)),
            A.Blur(p=1, blur_limit=(3, 5)),
            A.RandomToneCurve(p=1),
            A.RandomFog(p=1),
            A.CLAHE(p=1),
            A.Sharpen(p=1),
        ])
        
        for path in tqdm(self.img_path):
            img = cv2.imread(path)
            img = transform(image=img)['image']
            cv2.imwrite(os.path.join(self.argument_output,f"{self.enhance_time}" + os.path.basename(path)), img)

        print("addCompose done")

def match_label(label_path: str, label_type: str="txt",
                output_path: str=None, enhance_time: int=0) -> None:

    if output_path is not None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    original_labels = fetch_specific_files(label_path, "*")

    for index in tqdm(range(len(original_labels))):
        shutil.copyfile(original_labels[index], os.path.join(output_path, f"{enhance_time}" + os.path.splitext(os.path.basename(original_labels[index]))[0] + f".{label_type}" ))

    print("match_label done")
