import PIL.Image
import numpy as np
from typing import Union
from glob import glob
import pandas as pd
import os
from treys import Card # PROBLEM CAME UP AFTER INSTALLING PLOTLY
from termcolor import colored
from utils import eval_listof_games , debug_listof_games, save_results , load_results

import skimage.io
import matplotlib.pyplot as plt
#libraries for exercise 1.2 Region growing
from skimage.segmentation import flood, flood_fill
from skimage import morphology
from skimage.morphology import closing, opening, disk, square
import numpy as np

#libraries for exercise 1.3 Contour detection
from skimage import filters
import scipy
import cv2 as cv
import plotly.express as px

card_titles = ['Kcard', 'Qcard', 'Jcard', '10card', '9card', '8card', '7card', '6card', '5card', '4card', '3card', '2card', 'Acard']
ground_truth_titles = ['K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2', 'A', 'trèfle', 'pic', 'carreau', 'coeur']
    

def load_data(paths_list):
    images = []
    for path in paths_list:
        img = skimage.io.imread(path)
        images.append(img)
    return images


def isolate_card_features(cards, kings):
    # isolate cards
    idx = [520,1190,1190,1190,1190,1840,1840,1840,1890,2530,2530,2550,2550]
    row = 640
    idy = [2750,1940,2460,2980,3480,1915,2450,2950,3500,1930,2470,2975,3500]
    col = 470
    individual_cards = []

    for x,y in zip(idx, idy):
        card = cards[x:x+row, y:y+col]
        individual_cards.append(card)
    # isolate numbers/letters
    idx = [50,50,50,45,50,35,50,50,40,40,45,45,49]
    row = 70
    idy = [40,40,35,35,40,50,40,35,40,50,40,40,46]
    col = 50
    numbers = []

    for x,y,card in zip(idx, idy, individual_cards):
        number = card[x:x+row, y:y+col]
        numbers.append(number)
    # isolate symbols
    idx = [1620,1610,2280,2270]
    row = 60
    #idy = [2575,3088,2592,3140]
    idy = [2520,3033,2538,3090]
    col = 45
    symbols = []

    for x,y in zip(idx, idy):
        symbol = kings[x:x+row, y:y+col]
        symbols.append(symbol)

    return individual_cards, numbers, symbols
    

def one_contour_by_img(img_list):
    """
    Extracts the biggest contour for each image
    return: list(contour_img_1, contour_img_2, ...)
    """
    cont_img = []
    
    #Contours of images
    for img in img_list:
        # Compute all contours in image
        contours, hierarchy = cv.findContours(img.astype(np.uint8) ,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Case 1: 1 contour detected in image
        if len(contours) == 1:
            contour = contours[0][:,0]

        # Case 2: more than 1 contour detected
        else:
            # find and keep biggest contour, as others are artefacts
            big_contour_idx = 0
            for j in range(len(contours)):
                if len(contours[j]) > contours[big_contour_idx].shape[0]:
                    big_contour_idx = j
            contour = contours[big_contour_idx][:,0]

        # Record contour
        cont_img.append(contour)
    return cont_img


def contours_by_img(img_list):
    """ 
    Extracts all contours for each image
    return: list(  list(contour_1_img_1, ..., contour_m_img_1)  ,  ...,  list(contour_1_img_n, ..., contour_m_img_n)  )
    """
    cont_img = []
    
    #Contours of images
    for img in img_list:
        # Compute all contours in image
        contours, hierarchy = cv.findContours(img.astype(np.uint8) ,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        # Case 1: 1 contour detected in image
        if len(contours) == 1:
            contour = contours[0][:,0]
            contour = [contour]
            # append 1 contour in a list
            cont_img.append(contour)
            
            
        # Case 2: more than 1 contour detected
        else:
            # append all contours in a list (necessary reshaping)
            img_contours = [contours[i][:,0] for i in range(len(contours))]
            cont_img.append(img_contours)
        
    return cont_img

  
def filter_contours_by_size(sobel_all_contours, lower_bound, upper_bound):
    all_filtered_contours = []
    for contours in sobel_all_contours:
        image_conts = []
        for contour in contours:
            size = len(contour)
            if size > lower_bound and size < upper_bound:
                image_conts.append(contour)
        all_filtered_contours.append(image_conts)
    return all_filtered_contours


def complex_contours(contour_list):
    contours = []
    for contour in contour_list:
        complex_contour = contour[:,0] + 1j * contour[:,1]
        contours.append(complex_contour)
    return contours

def n_FT_descr(complex_contours, n):
    imgs_fft = []
    for complex_contour in complex_contours:
        fft = np.fft.fft(complex_contour)[:n]
        norm = np.abs(fft)
        imgs_fft.append(norm)
    imgs_fft = np.asarray(imgs_fft, dtype=object)
    return imgs_fft

def predict_cards_from_predictors(cards_3D_descr, GT_3D_descr, number_keys, symbol_keys):
    pred_numbers = []
    pred_symbols = []
    for card_all_descr in cards_3D_descr:
        # for each number, compute minimal distance to it
        dist_to_numbers = []
        for i in range(GT_3D_descr.shape[0]-4):
            diff = card_all_descr - GT_3D_descr[i,:]
            dist = diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2
            dist_to_numbers.append(np.min(dist))
        idx = np.argmin(dist_to_numbers)
        pred_numbers.append(number_keys[idx])
        
        # for each symbol, compute minimal distance to it
        dist_to_symbols = []
        for i in range(GT_3D_descr.shape[0]-4, GT_3D_descr.shape[0], 1):
            diff = card_all_descr - GT_3D_descr[i,:]
            dist = diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2
            dist_to_symbols.append(np.min(dist))
        idx = np.argmin(dist_to_symbols)
        pred_symbols.append(symbol_keys[idx])
        
    return pred_numbers, pred_symbols

def plot_coutours_length_distrib(cards_contours_len, ground_truth_contours_len):
    # plot ground truth distrib
    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(ground_truth_contours_len, bins=len(ground_truth_contours_len))
    plt.text(400,3.5, f'min = {np.min(ground_truth_contours_len)} \nmax = {np.max(ground_truth_contours_len)}')
    for i, (x, card) in enumerate(zip(ground_truth_contours_len, ground_truth_titles)):
        plt.text(x, (i+1)*0.2, card)
    plt.title('Distribution of ground truth contour lengths')
    plt.show()


    # plot cards distrib
    fig, axes = plt.subplots(ncols = 3, nrows = 5, figsize=(20,20))
    for ax, contours, title in zip(axes.flatten(), cards_contours_len, card_titles):
        lengths = [len(ct) for ct in contours]
        ax.hist(lengths, bins=len(lengths))
        ax.set_title(title)
    axes[-1,-1].axis('off')
    axes[-1,-2].axis('off')
    fig.suptitle('Distribution of contour length on cards')
    plt.show()

    
def plot_fourier_descr_and_card_contours(GT_descr, cards_descr = None, card_idx = None):
    fig, axes = plt.subplots(ncols = 2, figsize=(10,5))

    # CHOOSE WHICH CARD CONTOUR TO PLOT ON FOURIER SPACE
    if card_idx is not None and cards_descr is not None:
        j = card_idx
        axes[0].scatter(cards_descr[j][:,0], cards_descr[j][:,1],
                        color = 'k', label = card_titles[j], alpha = 0.3)
        axes[1].scatter(cards_descr[j][:,0], cards_descr[j][:,1],
                        color = 'k', label = card_titles[j], alpha = 0.3)

    # PLOT GROUND TRUTH FOURIER COEFS
    for descr, label in zip(GT_descr[:-4], ground_truth_titles[:-4]):
        axes[0].scatter(descr[0], descr[1], label = label)

    axes[0].legend(bbox_to_anchor=(1,1))
    axes[0].set_xlabel('1st descriptor')
    axes[0].set_ylabel('2nd descriptor')
    axes[0].set_title('Numbers/Letters Fourier descriptors')

    for descr, label in zip(GT_descr[-4:], ground_truth_titles[-4:]):
        axes[1].scatter(descr[0], descr[1], label = label)
    axes[1].legend(bbox_to_anchor=(1,1))
    axes[1].set_xlabel('1st descriptor')
    #axes[1].set_ylabel('2nd descriptor')
    axes[1].set_title('Symbols Fourier descriptors')
    plt.show()
    
def plot_fourier_descr_3D(GT_descr): 
    ax = plt.axes(projection='3d')
    for descr, label in zip(GT_descr[:-4], ground_truth_titles[:-4]):
        ax.scatter3D(descr[0], descr[1], descr[2], label = label)
    ax.set_title('Letters/Numbers 3D Fourier space')
    ax.set_xlabel('1st descriptor')
    ax.set_ylabel('2nd descriptor')
    ax.set_zlabel('3rd descriptor')
    ax.legend(bbox_to_anchor=(2,1))
    plt.show()

    ax = plt.axes(projection='3d')
    for descr, label in zip(GT_descr[-4:], ground_truth_titles[-4:]):
        ax.scatter3D(descr[0], descr[1], descr[2], label = label)
    ax.set_title('Symbols 3D Fourier space')
    ax.set_xlabel('1st descriptor')
    ax.set_ylabel('2nd descriptor')
    ax.set_zlabel('3rd descriptor')
    ax.legend(bbox_to_anchor=(2,1))
    plt.show()
    
def plot_interactive_3D_descr(df):
    fig = px.scatter_3d(df[:-4], x='descr 1', y='descr 2', z='descr 3')
    #fig.write_html('first_figure.html', auto_open=False)
    fig.show()
    fig = px.scatter_3d(df[-4:], x='descr 1', y='descr 2', z='descr 3')
    #fig.write_html('first_figure.html', auto_open=False)
    fig.show()

# def plot_card(idx, idy, r, c):
#     fig, axes = plt.subplots(ncols=4, figsize=(10,8))
#     for ax,x,y in zip(axes.flatten(),idx, idy):
#         ax.imshow(cards[x:x+r,y:y+c])
#         ax.axis('off')
#     plt.show()