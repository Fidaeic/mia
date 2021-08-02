
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from indices import *
from rasterio.plot import show
import seaborn as sns

from statsmodels.multivariate.pca import PCA as PCA_st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_som.som import SOM

np.seterr(divide='ignore')

def calc_indices(paths_dict, extension):
    b2 = rasterio.open(paths_dict['b2']).read(1)[:extension, :extension]
    b3 = rasterio.open(paths_dict['b3']).read(1)[:extension, :extension]
    b4 = rasterio.open(paths_dict['b4']).read(1)[:extension, :extension]
    b8 = rasterio.open(paths_dict['b8']).read(1)[:extension, :extension]
    b9 = rasterio.open(paths_dict['b9']).read(1)[:extension, :extension]
    b11 = rasterio.open(paths_dict['b11']).read(1)[:extension, :extension]
    b12 = rasterio.open(paths_dict['b12']).read(1)[:extension, :extension]
    rgb = rasterio.open(paths_dict['rgb']).read()[:, :extension, :extension]

    ndvi_ar = ndvi(b8, b4)
    gndvi_ar = gndvi(b8, b3)
    avi_ar = avi(b8, b4)
    savi_ar = savi(b8, b4)
    ndmi_ar = ndmi(b8, b11)
    msi_ar = msi(b8, b11)
    gci_ar = gci(b9, b3)
    nbri_ar = nbri(b8, b12)
    bsi_ar = bsi(b2, b4, b8, b11)
    evi_ar = evi(b8, b4, b2)
    ndwi_ar = ndwi(b8, b3)
    
    return np.stack([b2, b3, b4, b8, b9, b11, b12, ndvi_ar, gndvi_ar, avi_ar, \
                      savi_ar, ndmi_ar, msi_ar, gci_ar, nbri_ar, bsi_ar, evi_ar, ndwi_ar, rgb[0], rgb[1], rgb[2]], axis=2)


def window(X, window_size):
    
    
    # depth = X.shape[2] 
    nrows = X.shape[0]-2
    ncols = X.shape[1]-2
    
    # new_cols = window_size * depth
    new_matrix = np.empty(shape=(nrows*ncols, window_size))
    p = 0
    for i in range(1, X.shape[0]-1):
        for j in range(1, X.shape[1]-1):
            
            mask = X[i-1: i+2, j-1:j+2]
            
            vector = np.reshape(mask, newshape=mask.shape[0]*mask.shape[1])
    
            new_matrix[p, :] = vector
            
            p+=1
    return new_matrix

def window_append(X, window_size):
    
    depth = X.shape[2]
    
    final_matrix = np.array([])
    
    for d in range(depth):
        
        wind = window(X[:, :, d], window_size)
        
        if final_matrix.shape[0]==0:
            final_matrix = wind
        else:
            final_matrix = np.append(final_matrix, wind, axis=1)
    
    return final_matrix


def reduce(array, ncomps):
    pca = PCA_st(data=array, ncomp=ncomps, standardize=True, demean=True, normalize=True, method='nipals')

    print(pca.rsquare)

    # return im_clusters, final, resh, clusters
    return pca.scores, pca.loadings

def score_map(scores, component):
    
    shape = int(np.sqrt(scores.shape[0]))
    score_plot = np.reshape(scores[:, component], newshape=(shape, shape))

    plt.imshow(score_plot, interpolation='bilinear', cmap='RdBu')
    plt.colorbar()
    plt.title(f"Score map for component {component+1}")

    plt.show()
    
def loadings_plot(loadings, component):
    
    P_t = np.transpose(loadings)
    
    x = range(P_t.shape[1])
    
    plt.bar(x=x, height=P_t[component, :])

    
    plt.title(f"Loadings plot for component {component+1}")
    
    plt.show()
    
def scatter_scores(scores, component_1, component_2):
    
    plt.scatter(scores[:, component_1], scores[:, component_2])
    
    plt.axhline(color='red')
    plt.axvline(color='red')
    
    plt.title(f"Score plot for components {component_1+1} and {component_2+1}")
    
    plt.show()
    
def plot_original(extension):
    
    src = rasterio.open('../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_TCI_60m.jp2')
    
    image = src.read()[:, :extension, :extension]
    
    show(image)


def cluster(array):
    
    knn = KMeans()
    
    knn.fit(array)
    
    return knn.labels_


def save_as_tif(array, write_img, extension, component=None, scores=True):
    
    if scores==True:
        shape = int(np.sqrt(array.shape[0]))
        array = np.reshape(array[:, component], newshape=(shape, shape))
    
    
    src = rasterio.open(write_img)
    
    profile = src.profile.copy()
    
    profile.update(width=extension,
                   height=extension,
                   dtype=rasterio.uint8,
                   count=1,
                   compress='lzw')  
    
    with rasterio.open(write_img, 'w', **profile) as dst:
        dst.write(array.astype(rasterio.uint8), 1)
        

def ml(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    rf = RandomForestClassifier()
    
    rf.fit(X_train, y_train)
    
    return rf.score(X_test, y_test)

if __name__ == '__main__':
    PATH_b2 = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_B02_60m.jp2'
    PATH_b3 = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_B03_60m.jp2'
    PATH_b4 = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_B04_60m.jp2'
    PATH_b8 = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_B8A_60m.jp2'
    PATH_b9 = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_B09_60m.jp2'
    PATH_b11 = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_B11_60m.jp2'
    PATH_b12 = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_B12_60m.jp2'
    PATH_write = './SOM.jp2'
    PATH_rgb = '../S2A_MSIL2A_20201030T110211_N0214_R094_T30SVG_20201030T135011.SAFE/GRANULE/L2A_T30SVG_A027976_20201030T110451/IMG_DATA/R60m/T30SVG_20201030T110211_TCI_60m.jp2'
    
    paths_dict = {'b2': PATH_b2,
                  'b3': PATH_b3,
                  'b4': PATH_b4,
                  'b8': PATH_b8,
                   'b9': PATH_b9,
                  'b11': PATH_b11,
                  'b12': PATH_b12,
                  'rgb': PATH_rgb}
    
    ext = 900
    original = calc_indices(paths_dict, ext)

    
    plot_original(ext)

    wind = window_append(original, 9)
    
    pr = np.where(np.logical_or(np.isinf(wind), np.isnan(wind)), -9999, wind)


    T, P = reduce(pr, 50)
    
    score_map(T, 3)

    scatter_scores(T, 0, 1)
    
    loadings_plot(P, 0)
    
    
    labs = np.reshape(cluster(T), newshape=(T.shape[0], 1))
    
    score_map(labs, 0)
    
    
    
    
    
    
    
    
    
    save_as_tif(labs, './Clusters.jp2', ext, component=0, scores=True)
    
    
    
    som = SOM(m=21, n=10, dim=10)
    
    som.fit(T)
    
    pred = som.predict(T)

    im_pred = np.reshape(pred, newshape=(ext-2, ext-2))
    
    plt.imshow(im_pred)
    
    save_as_tif(im_pred, PATH_write, ext)
