import os
import cv2
import numpy as np


def all_files_under(path, extension=None, special=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if special not in fname]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)
                         if (special not in fname) and (fname.endswith(extension))]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if special not in fname]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)
                         if (special not in fname) and (fname.endswith(extension))]

    if sort:
        filenames = sorted(filenames)

    return filenames


def main(path_list, name_list):
    for idx_list, path in enumerate(path_list):
        gan_8gaussian_paths = all_files_under(os.path.join('img', path[0]), extension='jpg', special='disc')
        gan_25gaussian_paths = all_files_under(os.path.join('img', path[1]), extension='jpg', special='disc')
        gan_swissroll_paths = all_files_under(os.path.join('img', path[2]), extension='jpg', special='disc')

        wgan_8gaussian_paths = all_files_under(os.path.join('img', path[3]), extension='jpg', special='disc')
        wgan_25gaussian_paths = all_files_under(os.path.join('img', path[4]), extension='jpg', special='disc')
        wgan_swissroll_paths = all_files_under(os.path.join('img', path[5]), extension='jpg', special='disc')

        wgan_gp_8gaussian_paths = all_files_under(os.path.join('img', path[6]), extension='jpg', special='disc')
        wgan_gp_25gaussian_paths = all_files_under(os.path.join('img', path[7]), extension='jpg', special='disc')
        wgan_gp_swissroll_paths = all_files_under(os.path.join('img', path[8]), extension='jpg', special='disc')

        frame_shape = cv2.imread(wgan_8gaussian_paths[0]).shape
        print(frame_shape)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(name_list[idx_list], fourcc, 10.0, (frame_shape[1]*3, frame_shape[0]*3))

        for idx in range(len(wgan_8gaussian_paths)):
            img_gan8 = cv2.imread(gan_8gaussian_paths[idx])
            img_gan25 = cv2.imread(gan_25gaussian_paths[idx])
            img_gans = cv2.imread(gan_swissroll_paths[idx])

            img_wgan8 = cv2.imread(wgan_8gaussian_paths[idx])
            img_wgan25 = cv2.imread(wgan_25gaussian_paths[idx])
            img_wgans = cv2.imread(wgan_swissroll_paths[idx])

            img_wgangp8 = cv2.imread(wgan_gp_8gaussian_paths[idx])
            img_wgangp25 = cv2.imread(wgan_gp_25gaussian_paths[idx])
            img_wgangps = cv2.imread(wgan_gp_swissroll_paths[idx])

            frame_1 = np.hstack([img_gan8, img_gan25, img_gans])
            frame_2 = np.hstack([img_wgan8, img_wgan25, img_wgans])
            frame_3 = np.hstack([img_wgangp8, img_wgangp25, img_wgangps])
            frame = np.vstack([frame_1, frame_2, frame_3])

            # write the frame
            video_writer.write(frame)

            cv2.imshow('Show', frame)
            cv2.waitKey(1)

        # Release everything if job is finished
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    path01 = ['gan_8gaussians_True', 'gan_25gaussians_True', 'gan_swissroll_True',
              'wgan_8gaussians_True', 'wgan_25gaussians_True', 'wgan_swissroll_True',
              'wgan-gp_8gaussians_True', 'wgan-gp_25gaussians_True', 'wgan-gp_swissroll_True']
    path02 = ['gan_8gaussians_False', 'gan_25gaussians_False', 'gan_swissroll_False',
              'wgan_8gaussians_False', 'wgan_25gaussians_False', 'wgan_swissroll_False',
              'wgan-gp_8gaussians_False', 'wgan-gp_25gaussians_False', 'wgan-gp_swissroll_False']
    file_names = ['generator_fixed_true.mp4', 'generator_fixed_false.mp4']

    main([path01, path02], file_names)
