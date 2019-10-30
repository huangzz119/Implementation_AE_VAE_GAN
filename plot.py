import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import cv2

# select n image: comparing the original image and the construction image 
def plot_select(z_points, example_images, reconst_images, rand_selec_n_to_show):
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(rand_selec_n_to_show):
        img = example_images[i].squeeze()
        ax = fig.add_subplot(2, rand_selec_n_to_show, i+1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_points[i],1)), fontsize=10, ha='center', transform=ax.transAxes)   
        ax.imshow(img, cmap='gray_r')

    for i in range(rand_selec_n_to_show):
        img = reconst_images[i].squeeze()
        ax = fig.add_subplot(2, rand_selec_n_to_show, i+rand_selec_n_to_show+1)
        ax.axis('off')
        ax.imshow(img, cmap='gray_r')

# the distribution in the latent space
def plot_latent_distribution(z_points, example_labels):
    plt.figure(figsize=(6, 6))
    plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
                , alpha=0.5, s=2)
    plt.colorbar()
    plt.show()

        
def plot_embedding(X_2d, X_image, label):
    x_min, x_max = np.min(X_2d, 0), np.max(X_2d, 0)
    X_2d = (X_2d - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    for i in range(X_2d.shape[0]):
        plt.text(X_2d[i, 0], X_2d[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X_2d.shape[0]):
            dist = np.sum((X_2d[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X_2d[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(cv2.resize(X_image[i], (8,8)), cmap=plt.cm.gray_r),
                X_2d[i])
            ax.add_artist(imagebox)  
    plt.xticks([]), plt.yticks([])

        
def plot_latentspace(z_points, example_labels, decoder):
    plt.figure(figsize=(6, 6))
    plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
                , alpha=0.5, s=2)
    plt.colorbar()

    # x = norm.ppf(np.linspace(0.05, 0.95, 10))
    # y = norm.ppf(np.linspace(0.05, 0.95, 10))
    x = np.linspace(min(z_points[:, 0]), max(z_points[:, 0]), 6)
    y = np.linspace(max(z_points[:, 1]), min(z_points[:, 1]), 6)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    z_grid = np.array(list(zip(xv, yv)))

    reconst = decoder.predict(z_grid)
    plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'black'#, cmap='rainbow' , c= example_labels
                , alpha=1, s=5)
    plt.show()
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(6**2):
        ax = fig.add_subplot(6, 6, i+1)
        ax.axis('off')
        ax.imshow(reconst[i, :,:,0], cmap = 'Greys')
        
        
def Plot_sampling(z_points, decoder):
    plt.figure(figsize=(6, 6))
    plt.scatter(z_points[:, 0] , z_points[:, 1], c='black', alpha=0.5, s=2)

    grid_size = 10
    grid_depth = 3
    figsize = 15

    x = np.random.uniform(-5,5, size = grid_size * grid_depth)
    y = np.random.uniform(-5,5, size = grid_size * grid_depth)
    z_grid = np.array(list(zip(x, y)))
    reconst = decoder.predict(z_grid)

    plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'red', alpha=1, s=20)
    plt.show()

    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size*grid_depth):
        ax = fig.add_subplot(grid_depth, grid_size, i+1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_grid[i],1)), fontsize=10, ha='center', transform=ax.transAxes)

        ax.imshow(reconst[i, :,:,0], cmap = 'Greys')