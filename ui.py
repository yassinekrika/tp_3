from tkinter import ttk 
from tkinter import filedialog
import customtkinter
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np


class App(customtkinter.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Huffman Coding")
        self.geometry("800x500")
        self.maxsize(800, 500)
        self.minsize(800, 500)
        self.config(bg="#17043d")

        font1 = ("Arial", 20)

        self.image_path = ''
        
        self.code_button = customtkinter.CTkButton(self, text="select file",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.select_file)
        self.code_button.place(x=20, y=20)

        self.code_button = customtkinter.CTkButton(self, text="show image",font=font1, fg_color="#FF0000", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.show_img)
        self.code_button.place(x=300, y=20)

        self.code_button = customtkinter.CTkButton(self, text="image to YCrCb",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.image_to_ycrcb)
        self.code_button.place(x=20, y=60)

        self.code_button = customtkinter.CTkButton(self, text="YCrCb to image",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.ycrcb_to_rgb)
        self.code_button.place(x=20, y=100)

        self.code_button = customtkinter.CTkButton(self, text="4:2:2",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.sous_ech422)
        self.code_button.place(x=20, y=140)

        self.code_button = customtkinter.CTkButton(self, text="4:1:1",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.sous_ech411)
        self.code_button.place(x=20, y=180)

        self.code_button = customtkinter.CTkButton(self, text="DCT Formula",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.dct_formula)
        self.code_button.place(x=20, y=220)

        self.code_button = customtkinter.CTkButton(self, text="Inverse DCT Formula",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.inverse_dct_formula)
        self.code_button.place(x=20, y=260)

        self.code_button = customtkinter.CTkButton(self, text="Inverse DCT Matrix",font=font1, fg_color="#03a819", hover_color="#03a819", width=120, corner_radius=20, text_color="#ffffff", command=self.dct_matrix)
        self.code_button.place(x=20, y=300)

    def select_file(self):
            self.image_path = filedialog.askopenfilename(title='Open a file', initialdir='/home/yassg4mer/Project/tp_tmn/echentillonnage/')
            print(self.image_path)

    def image_to_ycrcb(self):
        
        image = cv2.imread(self.image_path)
        # Img to YCrCb
        R, G, B = cv2.split(image)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 128 + (-0.169 * R - 0.331 * G + 0.5 * B)
        Cr = 128 + (0.5 * R - 0.419 * G - 0.081 * B)

        ycrcb_image = cv2.merge((Y, Cr, Cb))
        

        cv2.imwrite("/home/yassg4mer/Project/tp_tmn/echentillonnage/YcrCb.bmp", ycrcb_image)

        # show figure
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.imshow(Y)
        plt.title('Y')
        plt.subplot(132)
        plt.imshow(Cr)
        plt.title('Cr')
        plt.subplot(133)
        plt.imshow(Cb)
        plt.title('Cb')
        plt.show()

        return ycrcb_image
        
    def ycrcb_to_rgb(self):
        ycrcb_image = self.image_to_ycrcb()
        Y, Cr, Cb = cv2.split(ycrcb_image)

        # Calculate the R, G, and B channels
        R = Y + 1.403 * (Cr - 128)
        G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
        B = Y + 1.773 * (Cb - 128)

        # Stack the R, G, and B channels to form the RGB image
        rgb_image = cv2.merge((R, G, B))

        cv2.imwrite("/home/yassg4mer/Project/tp_tmn/echentillonnage/RGB.bmp", rgb_image)

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

    # dct
    def matrix_to_blocks(self, matrix):
        blocks = []
        for i in range(0, matrix.shape[0], 8):
            for j in range(0, matrix.shape[1], 8):
                block = matrix[i:i+8, j:j+8]
                blocks.append(block)
        return blocks
    
    def c(k):
        if (k == 0):
            return np.sqrt(1/8) 
        else:
            return np.sqrt(2/8)

    def dct_block(self, image):
        M, N = image.shape

        dct_image = np.zeros((M, N))
        
        for u in range(M):
            for v in range(N):
                sum = 0
                for x in range(M):
                    for y in range(N):
                        cos_x = np.cos((2*x + 1) * u * np.pi / (2*M))
                        cos_y = np.cos((2*y + 1) * v * np.pi / (2*N))

                        sum += image[x,y] * cos_x * cos_y


                dct_image[u,v] = self.c(u) * self.c(v) * sum
        
        return dct_image
    

    def inverse_dct_block(self, dct_image):
        M, N = dct_image.shape
        image = np.zeros((M, N))
        
        for x in range(M):
            for y in range(N):
                sum = 0
                for u in range(M):
                    for v in range(N):
                        cos_x = np.cos((2*x + 1) * u * np.pi / (2*M))
                        cos_y = np.cos((2*y + 1) * v * np.pi / (2*N))

                        sum += self.c(u) * self.c(v) * dct_image[u,v] * cos_x * cos_y


                image[x,y] = sum
        
        return image

    def dct_formula(self):
        blocks = self.matrix_to_blocks(self.image_path)
        dct = []
        for block in blocks:
            dct.append(self.dct_block(block))

    def inverse_dct_formula(self):
        pass


    # dct matrix
    def dct_matrix(matrix):
        m = 8
        n = 8

        def c(k):
            if (k == 0):
                return 1 / (np.sqrt(m))
            else:
                return np.sqrt(2 / m)
            
        dct = np.zeros(m, n)
    
        for i in range(m):
            for j in range(n):

                if (i == 0 and j == 0):
                    ci = c(i)
                    cj = c(j)
                else:
                    ci = c(i)
                    cj = c(j)


                sum = 0
                for k in range(m):
                    for l in range(n):
    
                        sum += matrix[k][l] * np.cos((2 * k + 1) * i * np.pi / (
                            2 * m)) * np.cos((2 * l + 1) * j * np.pi / (2 * n))

    
                dct[i][j] = ci * cj * sum

        # test
        for i in range(m):
            for j in range(n):
                print(dct[i][j], end="\t")
            print()
 
    # show
    def show_img(self):
        img = cv2.imread("original_image.bmp", cv2.IMREAD_ANYCOLOR)
        plt.subplot(133)
        plt.imshow(img)
        plt.show()

if __name__=="__main__":
    app = App()
    app.mainloop()
