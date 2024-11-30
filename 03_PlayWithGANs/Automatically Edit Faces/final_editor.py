import os
import typing as tp
import uuid
import warnings
import gradio as gr
import torch
import numpy as np
import imageio
from draggan import utils
from draggan.draggan import drag_gan
from draggan import draggan as draggan
from face_alignment import FaceAlignment, LandmarksType
from numpy.typing import NDArray
from PIL import Image
import set_expressions  as exprs
from set_expressions import FaceExpression as Expr
import argparse

import torch.nn as nn


warnings.filterwarnings('ignore')

device = 'cuda'

SIZE_TO_CLICK_SIZE = {
    1024: 8,
    512: 5,
    256: 2
}

CKPT_SIZE = {
    'stylegan2/stylegan2-ffhq-config-f.pkl': 1024,
}

DEFAULT_CKPT = 'stylegan2/stylegan2-ffhq-config-f.pkl'


def get_landmarks(image: NDArray | Image.Image) -> NDArray:
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    fa = FaceAlignment(landmarks_type=LandmarksType.TWO_D)
    landmarks: list[NDArray] = fa.get_landmarks_from_image(image)
    return landmarks[0][:, ::-1]


def get_expression_points(
    landmarks: NDArray,
    expr_id: exprs.FaceExpression,
) -> list[list[float]]:
    start_pts, end_pts = exprs.transform(expr_id)(landmarks)
    return start_pts.tolist(), end_pts.tolist()


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')


def add_points_to_image(image, points, size=3):
    image = utils.draw_handle_target_points(image, points['handle'], points['target'], size)
    return image


def on_drag(model, points, max_iters, state, size, mask, lr_box):
    if len(points['handle']) == 0:
        raise gr.Error('You must select at least one handle point and target point.')
    if len(points['handle']) != len(points['target']):
        raise gr.Error('You have uncompleted handle points, try to select a target point or undo the handle point.')
    max_iters = int(max_iters)
    W = state['W']

    handle_points = [torch.tensor(p, device=device).float() for p in points['handle']]
    target_points = [torch.tensor(p, device=device).float() for p in points['target']]

    if mask.get('mask') is not None:
        mask = Image.fromarray(mask['mask']).convert('L')
        mask = np.array(mask) == 255
        mask = torch.from_numpy(mask).float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        mask = None

    step = 0
    for image, W, handle_points in drag_gan(W, model['G'], handle_points, target_points, mask, max_iters=max_iters, lr=lr_box):
        points['handle'] = [p.cpu().numpy().astype('int') for p in handle_points]
        image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
        state['history'].append(image)
        step += 1

    os.makedirs('final_tmp', exist_ok=True)
    image_name = f'final_tmp/image_{uuid.uuid4()}.png'
    video_name = f'final_tmp/video_{uuid.uuid4()}.mp4'
    imageio.imsave(image_name, image)
    imageio.mimsave(video_name, state['history'])
    return image, state, step


def on_max_iter_change(max_iters):
    return gr.update(maximum=max_iters)


def on_save_files(image, state):
    print(f'saving to {os.path.abspath("final_tmp")}')
    os.makedirs('final_tmp', exist_ok=True)
    image_name = f'final_tmp/image_{uuid.uuid4()}.png'
    video_name = f'final_tmp/video_{uuid.uuid4()}.mp4'
    imageio.imsave(image_name, image)
    imageio.mimsave(video_name, state['history'])
    return [image_name, video_name]


def on_show_save():
    return gr.update(visible=True)


def on_change_expression(
    expr_str: str,
    points: dict[str, tp.Any],
) -> dict[str, tp.Any]:
    landmarks: list[list[float]] = points['orig']
    expr_id: Expr = Expr.value_of(expr_str)
    points['handle'], points['target'] = get_expression_points(np.array(landmarks), expr_id)
    return points


def create_gradio_ui():
    torch.cuda.manual_seed(25)

    with gr.Blocks() as demo:
        G = draggan.load_model(utils.get_path(DEFAULT_CKPT), device=device)
        model = gr.State({'G': G})
        W = draggan.generate_W(
            G,
            seed=int(1),
            device=device,
            truncation_psi=0.8,
            truncation_cutoff=8,
        )
        img, F0 = draggan.generate_image(W, G, device=device)

        state = gr.State({
            'W': W,
            'img': img,
            'history': []
        })

        points_dict: dict[str, list[list[float]]] = {}
        landmarks: NDArray = get_landmarks(img)
        points_dict['orig'] = landmarks.tolist()
        points_dict['handle'], points_dict['target'] = get_expression_points(landmarks, Expr.CLOSE_EYES)
        points = gr.State(points_dict)

        size = gr.State(CKPT_SIZE[DEFAULT_CKPT])

        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Accordion('Drag'):
                    with gr.Row():
                        expr_menu = gr.Dropdown(choices=[
                            'Close eyes', 'Enlarge eyes', 'Close mouth', 'Smile mouth', 'Slim face',
                        ], value='Close eyes', label='Expression')

                        lr_box = gr.Number(value=5e-3, label='Learning Rate')
                        max_iters = gr.Slider(1, 500, 20, step=1, label='Max Iterations')

                    with gr.Row():
                        btn = gr.Button('Drag it', variant='primary')

                with gr.Accordion('Save', visible=False) as save_panel:
                    files = gr.Files(value=[])

                progress = gr.Slider(value=0, maximum=20, label='Progress', interactive=False)

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab('Setup Handle Points', id='input'):
                        image = gr.Image(img).style(height=512, width=512)
                    with gr.Tab('Draw a Mask', id='mask') as masktab:
                        mask = gr.ImageMask(img, label='Mask').style(height=512, width=512)

        expr_menu.change(fn=on_change_expression, inputs=[expr_menu, points], outputs=[points])
        btn.click(on_drag, inputs=[model, points, max_iters, state, size, mask, lr_box], outputs=[image, state, progress]).then(
            on_show_save, outputs=save_panel)
        max_iters.change(on_max_iter_change, inputs=max_iters, outputs=progress)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('-p', '--port', default=None)
    parser.add_argument('--ip', default=None)
    args = parser.parse_args()
    device = args.device
    demo = create_gradio_ui()
    print('Successfully loaded, starting gradio demo')
    demo.queue(concurrency_count=1, max_size=20).launch(share=args.share, server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()