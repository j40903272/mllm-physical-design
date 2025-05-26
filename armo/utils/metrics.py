from pytorch_msssim import SSIM


def validation(model, test_loader, device, ssim_fn):
    model.eval()
    val_ssim_score = 0.0
    with torch.no_grad():
        for image_tokens, gating_weights, multi_rewards, labels in test_loader:
            image_tokens, gating_weights, multi_rewards, labels = image_tokens.to(device), gating_weights.to(device), multi_rewards.to(device), labels.to(device)
            output = model(multi_rewards, gating_weights, image_tokens)
            val_ssim_score += ssim_fn(output, labels.unsqueeze(1))
            
    return val_ssim_score / len(test_loader)