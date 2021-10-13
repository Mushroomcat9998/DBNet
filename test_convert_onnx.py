# import onnx
# import torch.onnx
# from models import cropper
# from models import build_model
#
#
# device = torch.device('cpu')
#
#
# def static_onnx_converter(model_path, onnx_file):
#     # load checkpoint
#     # checkpoint = torch.load(model_path, map_location=device)
#     # # config for model architecture
#     # config = checkpoint['config']
#     # model = build_model(config['arch'])
#     # model.load_state_dict(checkpoint['state_dict'])
#     # model.to(device)
#     # model.eval()
#
#     model = cropper.U2NET()
#     if torch.cuda.is_available():
#         model.load_state_dict(torch.load(model_path))
#         model.to(torch.device("cuda"))
#     else:
#         model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     model.eval()
#
#     x = torch.rand(1, 3, 1024, 512, requires_grad=True)
#
#     print('Converting static model ...')
#     torch.onnx.export(model=model,  # model to be exported
#                       args=x,  # model input (or a tuple for multiple inputs)
#                       f=onnx_file,  # where to save onnx model (a file-like object)
#                       # export_params=True,  # store the trained parameter weights inside the model file
#                       opset_version=12,  # the ONNX version that the model is exported to (Default = 9)
#                       # main opset = 13; stable opsets = [7, 8, 9, 10, 11, 12]
#                       # do_constant_folding=True,  # whether to execute constant folding for optimization
#                       input_names=['input'],  # the model's input names
#                       output_names=['output'])  # the model's output names
#     print('Converted static onnx')
#
#
# def dynamic_onnx_converter(static_onnx_path,
#                            dynamic_onnx_path):
#     # save_folder = path.dirname(path.dirname(path.join(model_path)))
#     # static_onnx_path = path.join(save_folder, static_onnx_file)
#     # dynamic_onnx_path = path.join(save_folder, dynamic_onnx_file)
#
#     model = onnx.load(static_onnx_path)
#     model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
#     model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
#     model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
#
#     print('Converting dynamic model ...')
#     onnx.save(model, dynamic_onnx_path)
#     print('Converted dynamic onnx')
#
#
# # def init_args():
# #     parser = argparse.ArgumentParser(description='Convert_torch_to_onnx')
# #     parser.add_argument('--model_path',
# #                         # default='C:/Users/ADMIN/Desktop/u2net.pth',
# #                         default=r'D:\OCR\Localization\text_localization\models\detect_r34.pth',
# #                         type=str,
# #                         help='pytorch model path for conversion')
# #     # parser.add_argument('--onnx_path', default='model.onnx', type=str, help='onnx model file name after conversion')
# #     args = parser.parse_args()
# #     return args
#
#
# if __name__ == '__main__':
#     torch_file = 'C:/Users/ADMIN/Desktop/u2net.pth'
#     static_onnx_file = 'model.onnx'
#     dynamic_onnx_file = 'model_dynamic.onnx'
#     import gc
#     gc.collect()
#     # static_onnx_converter(model_path=torch_file,
#     #                       onnx_file=static_onnx_file)
#
#     dynamic_onnx_converter(static_onnx_path=static_onnx_file,
#                            dynamic_onnx_path=dynamic_onnx_file)
# # import gc
# # gc.collect()
# # np.testing.assert_allclose(to_numpy(torch_out), ort_outputs[0], rtol=1e-03, atol=1e-05)
