from tkinter import ttk 
from tkinter import filedialog
import customtkinter
import cv2
import matplotlib.pyplot as plt
import numpy as np


class App(customtkinter.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tp -3")
        self.geometry("800x500")
        self.maxsize(800, 500)
        self.minsize(800, 500)
        self.config(bg="#17043d")

        font1 = ("Arial", 20)

        self.image_path = self.image_path = filedialog.askopenfilename(title='Open a file', initialdir='/home/yassg4mer/Project/tp_tmn/echentillonnage/')
        self.dct_img = None

        self.label = customtkinter.CTkLabel(self, text="selected image : ", width=120, height=25, fg_color=("#17043d"), font=font1, text_color="#ffffff")
        self.label.place(x=5, y=5)

        self.path = customtkinter.CTkLabel(self, text=self.image_path, width=120, height=25, fg_color=("#17043d"),  text_color="#ffffff")
        self.path.place(x=5, y=30)

        self.code_button = customtkinter.CTkButton(self, text="image & YCrCb",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.image_and_ycrcb)
        self.code_button.place(x=20, y=80)

        self.code_button = customtkinter.CTkButton(self, text="sous echentillonnage",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.sous_ech411)
        self.code_button.place(x=20, y=160)

        self.code_button = customtkinter.CTkButton(self, text="DCT Formula",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.dct_formula)
        self.code_button.place(x=20, y=260)

        self.code_button = customtkinter.CTkButton(self, text="Inverse DCT Matrix",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.dct_matrix)
        self.code_button.place(x=20, y=340)

        self.code_button = customtkinter.CTkButton(self, text="psnr",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.call_psnr)
        self.code_button.place(x=20, y=400)

    def image_to_ycrcb(self):
        
        image = cv2.imread(self.image_path)
        # Img to YCrCb
        R, G, B = cv2.split(image)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 128 + (-0.169 * R - 0.331 * G + 0.5 * B)
        Cr = 128 + (0.5 * R - 0.419 * G - 0.081 * B)

        ycrcb_image = cv2.merge((Y, Cr, Cb))

        cv2.imwrite("/home/yassg4mer/Project/tp_tmn/echentillonnage/Ycrcb.bmp", ycrcb_image)
        return ycrcb_image
        
    def ycrcb_to_rgb(self, ycrcb_image):
        Y, Cr, Cb = cv2.split(ycrcb_image)

        # Calculate the R, G, and B channels
        R = Y + 1.403 * (Cr - 128)
        G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
        B = Y + 1.773 * (Cb - 128)

        # Stack the R, G, and B channels to form the RGB image
        rgb_image = cv2.merge((R, G, B))

        cv2.imwrite("/home/yassg4mer/Project/tp_tmn/echentillonnage/RGB.bmp", rgb_image)

        return rgb_image

    def image_and_ycrcb(self):
        ycrcb = self.image_to_ycrcb()
        rgb = self.ycrcb_to_rgb(ycrcb)
        
        y, cr, cb = cv2.split(ycrcb)

        # show figure ycrcb
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.imshow(y)
        plt.title('Y')
        plt.subplot(132)
        plt.imshow(cr)
        plt.title('Cr')
        plt.subplot(133)
        plt.imshow(cb)
        plt.title('Cb')
        plt.show()

        # show figure rgb
        rgb_img = cv2.imread("/home/yassg4mer/Project/tp_tmn/echentillonnage/RGB.bmp")
        plt.figure(figsize=(10,10))
        plt.subplot()
        plt.imshow(rgb_img)
        plt.title('original img')
        plt.show()

    def sous_ech422(self):
        ycrcb = self.image_to_ycrcb()

        y, cr, cb = cv2.split(ycrcb)

        new_cr_422 = np.zeros((len(cr), len(cr[0])))
        new_cb_422 = np.zeros((len(cr), len(cr[0])))

        # 4:2:2
        cr = cr[:, ::2]
        cb = cb[:, ::2]        

        new_cr_422 = np.repeat(cr, 2, axis=1)
        new_cb_422 = np.repeat(cb, 2, axis=1)  

        img422 = cv2.merge((y, new_cr_422, new_cb_422))
        
        cv2.imwrite("/home/yassg4mer/Project/tp_tmn/echentillonnage/SEch422.bmp", img422)

    def sous_ech411(self):
        ycrcb = self.image_to_ycrcb()
        
        y, cr, cb = cv2.split(ycrcb)

        new_cr_411 = np.zeros((len(cr), len(cr[0])))
        new_cb_411 = np.zeros((len(cr), len(cr[0])))

        # 4:1:1
        cr = cr[:, ::4]
        cb = cb[:, ::4]


        new_cr_411 = np.repeat(cr, 4, axis=1)
        new_cb_411 = np.repeat(cb, 4, axis=1)

        img411 = cv2.merge((y, new_cr_411, new_cb_411))
        cv2.imwrite("/home/yassg4mer/Project/tp_tmn/echentillonnage/SEch411.bmp", img411)
    
    def sous_ech(self):
        self.sous_ech422()
        self.sous_ech411()

    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        print('peack signal noise ration (PSNR) :', psnr)

    def call_psnr(self):
        img1 = filedialog.askopenfilename(title='img1', initialdir='/home/yassg4mer/Project/tp_tmn/echentillonnage/')
        img2 = filedialog.askopenfilename(title='img2', initialdir='/home/yassg4mer/Project/tp_tmn/echentillonnage/')

        self.psnr(img1, img2)

    # dct
    def dct(self, img):
        M, N = img.shape
        dct_img = np.zeros((M, N))

        # Compute constants c(u) and c(v)
        c = np.ones((8, 8))
        c[0, :] = 1 / np.sqrt(2)
        c[:, 0] = 1 / np.sqrt(2)

        # Compute 2D DCT
        for u in range(8, M + 1, 8):
            for v in range(8, N + 1, 8):
                img_block = img[u-8:u, v-8:v]
                dct_block = np.zeros((8, 8))
                for i in range(8):
                    for j in range(8):
                        sum_dct = 0.0
                        for x in range(8):
                            for y in range(8):
                                sum_dct += img_block[x, y] * np.cos((2 * x + 1) * i * np.pi / 8) * np.cos((2 * y + 1) * j * np.pi / 8)
                        dct_block[i, j] = 0.25 * c[i, j] * sum_dct
                dct_img[u-8:u, v-8:v] = dct_block

        return dct_img
    
    def inverse_dct(self, dct_img):
        M, N = dct_img.shape
        img = np.zeros((M, N))

        # Compute constants c(u) and c(v)
        c = np.ones((8, 8))
        c[0, :] = 1 / np.sqrt(2)
        c[:, 0] = 1 / np.sqrt(2)

        # Compute inverse 2D DCT
        for x in range(8, M + 1, 8):
            for y in range(8, N + 1, 8):
                dct_block = dct_img[x-8:x, y-8:y]
                img_block = np.zeros((8, 8))
                for i in range(8):
                    for j in range(8):
                        sum_idct = 0.0
                        for u in range(8):
                            for v in range(8):
                                sum_idct += c[u, v] * dct_block[u, v] * np.cos((2 * i + 1) * u * np.pi / 16) * np.cos((2 * j + 1) * v * np.pi / 16)
                        img_block[i, j] = 0.25 * sum_idct
                img[x-8:x, y-8:y] = img_block

        return img
    
    def dct_formula(self):
        img = cv2.imread('original_image.bmp', cv2.IMREAD_GRAYSCALE)
        img = np.subtract(img, 128)
        self.dct_img = self.dct(img)
        inverse_dct = self.inverse_dct(self.dct_img)


        cv2.imshow('Original Image', img)
        cv2.imshow('DCT Image', self.dct_img.astype('uint8'))
        cv2.imshow('Inverse dct image', inverse_dct.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def inverse_dct_formula(self):
        img_reconstructed = self.idct_2d(self.dct_img)
        cv2.imshow('Inverse dct image', img_reconstructed.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # dct matrix
    def dct_matrix(matrix):
        pass

if __name__=="__main__":
    app = App()
    app.mainloop()
