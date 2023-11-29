import os
import torch

def save_model(model, model_save_path, model_name='model'):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(model_save_path, 'model.pt'))  # save model.module.state_dict()

    # torch.save(model.module.state_dict(), os.path.join(model_save_path, 'model.pt'))


def load_model(args):
    model_path = os.path.join(args.model.model_save_path, 'model.pt')

    model = torch.load(model_path, map_location='cpu')
    if args.args.is_cuda:
        model = model.module.cuda()
    return model


def load_avg_model_state(args):
    model_path = os.path.join(args.ckpt_out_file, 'avg_model.pt')

    model_state = torch.load(model_path, map_location='cpu')
    return model_state
