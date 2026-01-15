def from_pretrained_2d(cls, pretrained_model_path, subfolder=None):
    if subfolder is not None:
        pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

    config_file = os.path.join(pretrained_model_path, 'config.json')
    if not os.path.isfile(config_file):
        raise RuntimeError(f"{config_file} does not exist")
        
    with open(config_file, "r") as f:
        config = json.load(f)
    config["_class_name"] = cls.__name__
    # 保持原有block类型配置
    
    model = cls.from_config(config)
    from diffusers.utils import WEIGHTS_NAME
    model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
    
    if not os.path.isfile(model_file):
        raise RuntimeError(f"{model_file} does not exist")
        
    state_dict = torch.load(model_file, map_location="cpu")
    
    # 新增：智能键映射处理
    key_mapping = {
        # 新模块到旧模块的映射（示例）
        "mid_block.patch_embed3.": "mid_block.patch_embed2.",
        "mid_block.norm3.": "mid_block.norm2.",
        "mid_block.kan_block1.0.": "mid_block.kan_block0.0."
    }
    
    # 应用键映射
    mapped_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for pat, repl in key_mapping.items():
            if pat in k:
                new_k = k.replace(pat, repl)
                break
        mapped_state_dict[new_k] = v
    
    # 合并原始state_dict和映射后的版本
    combined_dict = {**state_dict, **mapped_state_dict}
    
    # 新增：参数继承策略
    for name, module in model.named_modules():
        if "mid_block" in name:  # 针对新增模块区域
            if isinstance(module, nn.Conv3d) and "proj" in name:
                # 继承最近conv层的参数
                for parent_name, parent_module in model.named_modules():
                    if "mid_block" in parent_name and isinstance(parent_module, nn.Conv3d):
                        if hasattr(parent_module, 'weight'):
                            module.weight = parent_module.weight
                        if hasattr(parent_module, 'bias'):
                            module.bias = parent_module.bias
                        break
    
    # 关键修改：允许非严格加载
    model.load_state_dict(combined_dict, strict=False)  # 允许缺失键
    
    # 新增：初始化新增模块
    for name, module in model.named_modules():
        if "kan_block" in name or "patch_embed" in name:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    return model
