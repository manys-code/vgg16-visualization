"""
Visualize the learned filter for a pretrained VGG16 network
(Adapted from the https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html blogpost to Gluon)
"""
import os

from argparse import ArgumentParser
from collections import namedtuple

import matplotlib.pyplot as plt

from mxnet import autograd, gluon, nd
from skimage.io import imread

ConvLayer = namedtuple("ConvLayer", "index filters")

def loss(x_hat, filter) -> nd.NDArray:
    return nd.mean(x_hat[:, filter, :, :])

def std(x):
    return nd.sqrt(nd.sum(nd.square(x - x.mean())) / x.size)

def get_vgg16():
    vgg16 = gluon.model_zoo.vision.vgg16(pretrained=True).features[:31]
    layer_dict = {layer.name[5:]: ConvLayer(index, layer._channels) for index, layer in enumerate(vgg16) if "conv" in layer.name}
    return vgg16, layer_dict

def deprocess_image(img):
    """Revert normalization"""
    img -= img.mean()
    img /= (std(img) + 1e-5)
    img *= 0.1

    # clip
    img += 0.5
    img = nd.clip(img, 0, 1)
    img = img.transpose((1, 2, 0)).asnumpy()
    return img


def generate_visualization(subnet, input_image, filter, iterations, lr):
    input_image = input_image.copy()
    input_image.attach_grad()
    for _ in range(iterations):
        with autograd.record():
            x_hat = subnet(input_image)
            l = loss(x_hat, filter)
        l.backward()

        img_grads = input_image.grad
        # Normalize gradients and perform gradient ascent
        img_grads /= (nd.sqrt(nd.mean(nd.square(img_grads))) + 1e-5)
        input_image += img_grads * lr
    
    return input_image

def visualize_layer(nrows, ncols, maps, title, path, vis_size=3):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(vis_size * ncols, vis_size * nrows))
    axs = axs.ravel()

    for ax, map in zip(axs, maps):
        ax.imshow(deprocess_image(map))
        ax.axis("off")
    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(path, "{}.png".format(title)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize convolutional filters from a pretrained VGG16 net")
    parser.add_argument("--list", help="List convolutional layer names", action="store_true")
    parser.add_argument("--layer", help="Layer name. Use --list to list layer names in VGG16")
    parser.add_argument("--nrows", type=int, help="Number of rows to display visualizations")
    parser.add_argument("--ncols", type=int, help="Number of columns to display visualizations")
    parser.add_argument("--image", help="[Optional] Image to use in the visualization process. If no image is provided, a random image is generated")
    parser.add_argument("--lr", type=float, help="[Optional] Learning rate for gradient ascent. Default value is 0.1")
    parser.add_argument("--iterations", type=int, help="[Optional] Number of iterations to generate the visualization. Default value is 20")
    parser.add_argument("--path", help="[Optional ]Directory to store visualization. Standard name is '<layername>'. Default value is '.'")
    parser.add_argument("--all", help="Generate visualizations for all convolutional layers", action="store_true")

    vgg16, layer_dict = get_vgg16()

    args = parser.parse_args()

    if args.list:
        print("Convolutional layer names:")
        for layer, data in layer_dict.items():
            print("\t{}\t{} filters".format(layer, data.filters))
            exit()
    if args.image:
        input_image = nd.array(imread(args.image)).transpose((0, 1, 2))
        input_image = input_image.expand_dims(axis=0)
    else:
        input_image = nd.random.uniform(shape=(1, 3, 225, 225))

    nrows = args.nrows
    ncols = args.ncols
    lr = args.lr if args.lr else 0.1
    iterations = args.iterations if args.iterations else 20
    path = args.path if args.path else "."
    if not os.path.exists(path):
        os.makedirs(path)
    if args.all:
        for layer_name, target_layer in layer_dict.items():
            subnet = vgg16[:target_layer.index + 1]
            total_filters = nrows * ncols
            assert total_filters <= target_layer.filters
            visualizations = [generate_visualization(subnet, input_image, filter, iterations, lr)[0] for filter in range(total_filters)]
            visualize_layer(nrows, ncols, visualizations, layer_name, path)
    else:
        target_layer = layer_dict[args.layer]
        subnet = vgg16[:target_layer.index + 1]
        total_filters = nrows * ncols
        assert total_filters <= target_layer.filters
        visualizations = [generate_visualization(subnet, input_image, filter, iterations, lr)[0] for filter in range(total_filters)]
        visualize_layer(nrows, ncols, visualizations, args.layer, path)
    