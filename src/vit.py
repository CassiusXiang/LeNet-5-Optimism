import matplotlib.pyplot as plt

def vit_pic_from_train_loader(data_train_loader):
    data_iter = iter(data_train_loader)
    images, labels = next(data_iter)  # images 大小: [batch_size, channels, height, width]

    num_images_to_show = 6
    plt.figure(figsize=(10, 5))
    for i in range(num_images_to_show):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i][0].cpu().numpy(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    