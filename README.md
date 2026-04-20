# PixelCNN
PixelCNN is an autoregressive image model that factors p(x) = ∏ p(xᵢ | x_{&lt;i}) in raster order. It uses masked convolutions (Mask A in layer 1, Mask B after) to ensure each pixel only sees previous ones. Training is one forward pass with BCE loss; sampling is sequential, one pixel at a time. Has a blind spot in the upper-right region.
