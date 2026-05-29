import torch

def load_dataset_eps(args):
    if args.dataset == 'cifar10':
        if args.use_score:
            print("Using Score Norm as reweighting metric!")
            cf10_pt = f"Enter the path to the score file by running cifar10_eps_range.sh"
        else:
            # cf10_pt = f"Enter the path to the eps file by running cifar10_eps_range.sh"
            cf10_pt = "/data/gpfs/projects/punim2205/ICLR2025/DBP/loaded_data/cifar10/sd121/5000_cifar10_eps.pt"
        cf10_eps_data = torch.load(cf10_pt)

        cf10_nat_eps_list = []
        for score in cf10_eps_data:
            cf10_nat_eps_list.append(torch.norm(score.flatten(), p=2).cpu().item())

        cf10_eps_standard = cf10_nat_eps_list
        return cf10_eps_standard
    elif args.dataset == 'imagenet':
        # imgnet_pt = f"Enter the path to the eps file by running imagenet_eps_range.sh"
        imgnet_pt = "/data/gpfs/projects/punim2205/ICLR2025/DBP/loaded_data/imagenet/sd121/5000_imagenet_eps.pt"
        imgnet_eps_data = torch.load(imgnet_pt)

        imgnet_nat_eps_list = []
        for score in imgnet_eps_data:
            imgnet_nat_eps_list.append(torch.norm(score.flatten(), p=2).cpu().item())

        imagenet_eps_standard = imgnet_nat_eps_list
        return imagenet_eps_standard
    else:
        raise NotImplementedError("Unknown dataset.")