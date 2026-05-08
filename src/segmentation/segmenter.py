from .kmeans import kmeans_segment
from .watershed import watershed_segment


def segment(image, method="kmeans"):

    if method == "kmeans":

        segmented, labels, compactness = kmeans_segment(image)

        return {
            "segmented_image": segmented,
            "labels": labels,
            "compactness": compactness,
            "method": method
        }

    elif method == "watershed":

        segmented, labels = watershed_segment(image)
        #segmented, labels, masks = watershed_segment(image)


        return {
            "segmented_image": segmented,
            "labels": labels,
            #"masks": masks,
            "method": method
        }

    else:
        raise ValueError("Unknown segmentation method")