def from_pretrained_2d(cls, pretrained_model_path, subfolder=None):
    if subfolder is not None:
        pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

    config_file = os.path.join(pretrained_model_path, 'config.json')
    if not os.path.isfile(config_file):
        raise RuntimeError(f"{config_file} does not exist")
        
    with open(config_file, "r") as f:
        config = json.load(f)
    config["_class_name"] = cls.__name__
    config["down_block_types"] = [
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D"
    ]
    config["up_block_types"] = [
        "UpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D"
    ]
    
    from diffusers.utils import WEIGHTS_NAME
    model = cls.from_config(config)
    model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
    
    if not os.path.isfile(model_file):
        raise RuntimeError(f"{model_file} does not exist")
        
    state_dict = torch.load(model_file, map_location="cpu")
    
    # 保留原有的2D→3D键映射逻辑
    for k, v in model.state_dict().items():
        if '_temp.' in k or 'conv1d' in k or 'conv_gate' in k:
            state_dict.update({k: v})
        if 'conv2d.' in k:
            origin_k = k.replace('conv2d.', '')
            origin_v = state_dict.pop(origin_k)
            state_dict.update({k: origin_v})
            #if origin_k in state_dict:
            #    state_dict.update({k: state_dict.pop(origin_k)})
    
    # 非严格加载并捕获缺失键
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    
    # 零初始化新增模块（根据报错信息中的模块名模式）
    new_module_patterns = [
        "mid_block.patch_embed3",
        "mid_block.patch_embed4",
        "mid_block.norm3",
        "mid_block.norm4",
        "mid_block.dnorm3",
        "mid_block.kan_block",
        "mid_block.kan_dblock",
        "mid_block.decoder"
    ]
    
    # 遍历所有参数进行零初始化
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in new_module_patterns):
            param.data.zero_()  # 核心修改：零初始化新增模块
    
    # 可选：对特定层类型进行零初始化
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            if any(pattern in name for pattern in new_module_patterns):
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()  # 使用模块自带的初始化方法
                else:
                    module.weight.data.zero_()
                    if module.bias is not None:
                        module.bias.data.zero_()
    
    return model
