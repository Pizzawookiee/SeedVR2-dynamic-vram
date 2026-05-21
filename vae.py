class VideoAutoencoderKL(diffusers.AutoencoderKL):
    """
    We simply inherit the model code from diffusers
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        attention: bool = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size: int = 32,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        *args,
        **kwargs,
    ):
        extra_cond_dim = kwargs.pop("extra_cond_dim") if "extra_cond_dim" in kwargs else None
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = max(1, slicing_sample_min_size // (2**temporal_scale_num))

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            # [Override] make sure it can be normally initialized
            down_block_types=tuple(
                [down_block_type.replace("3D", "2D") for down_block_type in down_block_types]
            ),
            up_block_types=tuple(
                [up_block_type.replace("3D", "2D") for up_block_type in up_block_types]
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
            *args,
            **kwargs,
        )

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            extra_cond_dim=extra_cond_dim,
            # [Override] add temporal_down_num parameter
            temporal_down_num=temporal_scale_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.quant_conv = (
            init_causal_conv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            init_causal_conv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64)) -> AutoencoderKLOutput:
        if tiled:
            h = self.tiled_encode(x, tile_size=tile_size, tile_overlap=tile_overlap)
        else:
            h = self.slicing_encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64)) -> Union[DecoderOutput, torch.Tensor]:

        if tiled:
            decoded = self.tiled_decode(z, tile_size=tile_size, tile_overlap=tile_overlap)
        else:
            decoded = self.slicing_decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _encode(
        self, x: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> torch.Tensor:
        # Only transfer if not already on correct device
        _x = x if x.device == self.device else x.to(self.device)
        
        _x = causal_conv_slice_inputs(_x, self.slicing_sample_min_size, memory_state=memory_state)
        h = self.encoder(_x, memory_state=memory_state)
        
        if self.quant_conv is not None:
            output = self.quant_conv(h, memory_state=memory_state)
        else:
            output = h
        
        output = causal_conv_gather_outputs(output)
        
        # MPS memory leak workaround (pytorch/pytorch#155060)
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        
        # Only transfer back if needed
        return output if output.device == x.device else output.to(x.device)

    def _decode(
        self, z: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> torch.Tensor:
        # Only transfer if not already on correct device
        _z = z if z.device == self.device else z.to(self.device)
        
        _z = causal_conv_slice_inputs(_z, self.slicing_latent_min_size, memory_state=memory_state)
        
        if self.post_quant_conv is not None:
            _z = self.post_quant_conv(_z, memory_state=memory_state)
        
        output = self.decoder(_z, memory_state=memory_state)
        output = causal_conv_gather_outputs(output)
        
        # MPS memory leak workaround (pytorch/pytorch#155060)
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        
        # Only transfer back if needed
        return output if output.device == z.device else output.to(z.device)

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size * sp_size:
            x_slices = x[:, :, 1:].split(split_size=self.slicing_sample_min_size * sp_size, dim=2)
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE)
                )
            out = torch.cat(encoded_slices, dim=2)
            # Clear memory efficiently
            modules_with_memory = [m for m in self.modules() 
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            return self._encode(x)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size * sp_size:
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size * sp_size, dim=2)
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING
                )
            ]
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE)
                )
            out = torch.cat(decoded_slices, dim=2)
            # Clear memory efficiently
            modules_with_memory = [m for m in self.modules() 
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            return self._decode(z)

    def tiled_encode(self, x: torch.Tensor, tile_size: Tuple[int, int] = (512, 512), 
                     tile_overlap: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        r"""
        Encodes an input tensor `x` by splitting it into spatial tiles in latent space. Temporal is handled by `slicing_encode`.
        `tile_size` and `tile_overlap` are interpreted in output-space pixels and converted to latent-space.
        """
        # Ensure 5D [B, C, F, H, W]
        if x.ndim != 5:
            x = x.unsqueeze(2)

        b, c, f, H, W = x.shape
        tile_h, tile_w = tile_size
        
        # Only tile if input resolution requires multiple tiles
        if H <= tile_h and W <= tile_w:
            return self.slicing_encode(x)
        else:
            if self.debug:
                self.debug.log(f"Using VAE tiled encoding (Tile: {tile_size}, Overlap: {tile_overlap})", category="vae", force=True, indent_level=1)

        # Spatial scale factor (output/latent)
        scale_factor = self.spatial_downsample_factor

        # Convert output-space tiling params to latent-space
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap
        
        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)
        latent_overlap_h = max(0, min((overlap_h // scale_factor), latent_tile_h - 1))
        latent_overlap_w = max(0, min((overlap_w // scale_factor), latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        H_lat_total = (H + scale_factor - 1) // scale_factor
        W_lat_total = (W + scale_factor - 1) // scale_factor

        result = None
        count = None

        num_tiles = ((max(H_lat_total - latent_overlap_h, 1) + stride_h - 1) // stride_h) \
                  * ((max(W_lat_total - latent_overlap_w, 1) + stride_w - 1) // stride_w)

        # Log once at start instead of per-tile
        if self.debug:
            self.debug.log(
                f"Encoding {num_tiles} tiles (Tile: {tile_size}, Overlap: {tile_overlap})",
                category="vae",
            )

        # Pre-compute common ramp values
        ramp_cache = {}
        if latent_overlap_h > 0:
            t_h = torch.linspace(0, 1, steps=latent_overlap_h, device=x.device, dtype=x.dtype)
            ramp_cache['h'] = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
        if latent_overlap_w > 0:
            t_w = torch.linspace(0, 1, steps=latent_overlap_w, device=x.device, dtype=x.dtype)
            ramp_cache['w'] = 0.5 - 0.5 * torch.cos(t_w * torch.pi)

        tile_id = 0
        for y_lat in range(0, H_lat_total, stride_h):
            y_lat_end = min(y_lat + latent_tile_h, H_lat_total)
            for x_lat in range(0, W_lat_total, stride_w):
                x_lat_end = min(x_lat + latent_tile_w, W_lat_total)

                # Skip if fully within overlap of previous tiles
                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                    continue

                # Map latent tile to output-space crop
                y_out = y_lat * scale_factor
                x_out = x_lat * scale_factor
                y_out_end = min(y_lat_end * scale_factor, H)
                x_out_end = min(x_lat_end * scale_factor, W)

                tile_id += 1

                # Store tile boundary info for debug visualization
                if self.debug and hasattr(self.debug, 'encode_tile_boundaries'):
                    self.debug.encode_tile_boundaries.append({
                        'id': tile_id,
                        'y': y_out,
                        'x': x_out,
                        'h': y_out_end - y_out,
                        'w': x_out_end - x_out
                    })

                tile_sample = x[:, :, :, y_out:y_out_end, x_out:x_out_end]

                # Log progress periodically instead of every tile (at 1, 6, 11, 16, ...)
                if self.debug and (tile_id % 5 == 1 or tile_id == num_tiles):
                    if tile_id == num_tiles:
                        # Only log final tile if not covered by previous range
                        if (tile_id - 1) % 5 == 0:
                            self.debug.log(f"Encoding tile {tile_id} / {num_tiles}", category="vae", indent_level=1)
                    else:
                        end_tile = min(tile_id + 4, num_tiles)
                        self.debug.log(f"Encoding tiles {tile_id}-{end_tile} / {num_tiles}", category="vae", indent_level=1)

                encoded_tile = self.slicing_encode(tile_sample)

                # Initialize output size using first encoded tile
                if result is None:
                    b_out, c_out, f_lat, _, _ = encoded_tile.shape
                    
                    # Accumulate on offload device if specified and different, else on inference device
                    device = getattr(self, 'tensor_offload_device', None)
                    if device is None or device == encoded_tile.device:
                        device = encoded_tile.device
                    
                    result = torch.zeros(
                        (b_out, c_out, f_lat, H_lat_total, W_lat_total),
                        device=device,
                        dtype=encoded_tile.dtype,
                    )
                    count = torch.zeros((1, 1, 1, H_lat_total, W_lat_total), device=device, dtype=encoded_tile.dtype)

                eff_h_lat = min(y_lat_end - y_lat, encoded_tile.shape[3], result.shape[3] - y_lat)
                eff_w_lat = min(x_lat_end - x_lat, encoded_tile.shape[4], result.shape[4] - x_lat)

                encoded_tile = encoded_tile[:, :, : result.shape[2], :eff_h_lat, :eff_w_lat]

                # Build faded masks
                ov_h = max(0, min(latent_overlap_h, eff_h_lat - 1))
                ov_w = max(0, min(latent_overlap_w, eff_w_lat - 1))
                
                weight_h = torch.ones((eff_h_lat,), device=encoded_tile.device, dtype=encoded_tile.dtype)
                weight_w = torch.ones((eff_w_lat,), device=encoded_tile.device, dtype=encoded_tile.dtype)

                # Apply fades only on interior edges using cached ramps (avoid fading on outer image borders)
                if ov_h > 0:
                    if y_lat > 0:  # Not top edge
                        weight_h[:ov_h] = ramp_cache['h'][:ov_h]
                    if y_lat_end < H_lat_total:  # Not bottom edge
                        weight_h[-ov_h:] = 1 - ramp_cache['h'][:ov_h]
                if ov_w > 0:
                    if x_lat > 0:  # Not left edge
                        weight_w[:ov_w] = ramp_cache['w'][:ov_w]
                    if x_lat_end < W_lat_total:  # Not right edge
                        weight_w[-ov_w:] = 1 - ramp_cache['w'][:ov_w]

                # Separable application (no 2D mask to save memory)
                weight_h_5d = weight_h.view(1, 1, 1, eff_h_lat, 1)
                weight_w_5d = weight_w.view(1, 1, 1, 1, eff_w_lat)
                encoded_tile.mul_(weight_h_5d).mul_(weight_w_5d)

                # Accumulate (move to result device if different)
                if result.device != encoded_tile.device:
                    encoded_tile = encoded_tile.to(result.device)
                    weight_h_5d = weight_h_5d.to(result.device)
                    weight_w_5d = weight_w_5d.to(result.device)
                
                result[:, :, : encoded_tile.shape[2], y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat] += encoded_tile
                count[:, :, :, y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat].addcmul_(weight_h_5d, weight_w_5d)

        # Move result back to inference device if needed and normalize
        if result.device != x.device:
            result = result.to(x.device)
            count = count.to(x.device)
        result.div_(count.clamp(min=1e-6))

        if x.shape[2] == 1:  # single frame
            result = result.squeeze(2)

        return result

    def tiled_decode(self, z: torch.Tensor, tile_size: Tuple[int, int] = (512, 512), tile_overlap: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        r"""
        Decodes a latent tensor `z` by splitting it into spatial tiles only. Temporal is handled by `slicing_decode`.
        """
        if z.ndim != 5:
            z = z.unsqueeze(2)

        b, c, f, H, W = z.shape

        # Spatial scale factor (output/latent)
        scale_factor = self.spatial_downsample_factor

        # Convert output-space tiling params to latent-space for spatial tiling
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap
        
        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)
        
        # Only tile if latent resolution requires multiple tiles
        if H <= latent_tile_h and W <= latent_tile_w:
            return self.slicing_decode(z)
        else:
            if self.debug:
                self.debug.log(f"Using VAE tiled decoding (Tile: {tile_size}, Overlap: {tile_overlap})", category="vae", force=True, indent_level=1)
        
        latent_overlap_h = max(0, min((overlap_h // scale_factor), latent_tile_h - 1))
        latent_overlap_w = max(0, min((overlap_w // scale_factor), latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        # Allocate later using first decoded results
        result = None
        count = None

        num_tiles = ((max(H - latent_overlap_h, 1) + stride_h - 1) // stride_h) \
                  * ((max(W - latent_overlap_w, 1) + stride_w - 1) // stride_w)

        # Log once at start instead of per-tile
        if self.debug:
            self.debug.log(
                f"Decoding {num_tiles} tiles (Tile: {tile_size}, Overlap: {tile_overlap})",
                category="vae",
            )

        # Pre-compute common ramp values (small memory, big time save)
        ramp_cache = {}
        if overlap_h > 0:
            t_h = torch.linspace(0, 1, steps=overlap_h, device=z.device, dtype=z.dtype)
            ramp_cache['h'] = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
        if overlap_w > 0:
            t_w = torch.linspace(0, 1, steps=overlap_w, device=z.device, dtype=z.dtype)
            ramp_cache['w'] = 0.5 - 0.5 * torch.cos(t_w * torch.pi)

        tile_id = 0
        for y_lat in range(0, H, stride_h):
            y_lat_end = min(y_lat + latent_tile_h, H)
            for x_lat in range(0, W, stride_w):
                x_lat_end = min(x_lat + latent_tile_w, W)

                # Skip if fully within overlap of previous tiles
                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                    continue

                tile_id += 1
                
                # Store tile boundary info for debug visualization
                if self.debug and hasattr(self.debug, 'decode_tile_boundaries'):
                    # Map to output space
                    y_out = y_lat * scale_factor
                    x_out = x_lat * scale_factor
                    y_out_end = y_lat_end * scale_factor
                    x_out_end = x_lat_end * scale_factor
                    self.debug.decode_tile_boundaries.append({
                        'id': tile_id,
                        'y': y_out,
                        'x': x_out,
                        'h': y_out_end - y_out,
                        'w': x_out_end - x_out
                    })
                
                tile_latent = z[:, :, :, y_lat:y_lat_end, x_lat:x_lat_end]

                # Log progress periodically instead of every tile (at 1, 6, 11, 16, ...)
                if self.debug and (tile_id % 5 == 1 or tile_id == num_tiles):
                    if tile_id == num_tiles:
                        # Only log final tile if not covered by previous range
                        if (tile_id - 1) % 5 == 0:
                            self.debug.log(f"Decoding tile {tile_id} / {num_tiles}", category="vae", indent_level=1)
                    else:
                        end_tile = min(tile_id + 4, num_tiles)
                        self.debug.log(f"Decoding tiles {tile_id}-{end_tile} / {num_tiles}", category="vae", indent_level=1)

                decoded_tile = self.slicing_decode(tile_latent)

                # Initialize result tensors using actual decoded shapes on first tile
                if result is None:
                    b_out, c_out, out_f_tile, _, _ = decoded_tile.shape
                    output_h = H * scale_factor
                    output_w = W * scale_factor
                    
                    # Accumulate on offload device if specified and different, else on inference device
                    device = getattr(self, 'tensor_offload_device', None)
                    if device is None or device == decoded_tile.device:
                        device = decoded_tile.device
                    
                    result = torch.zeros((b_out, c_out, out_f_tile, output_h, output_w), device=device, dtype=decoded_tile.dtype)
                    count = torch.zeros((1, 1, 1, output_h, output_w), device=device, dtype=decoded_tile.dtype)

                # Corresponding output-space placement
                y_out, y_out_end = y_lat * scale_factor, y_lat_end * scale_factor
                x_out, x_out_end = x_lat * scale_factor, x_lat_end * scale_factor

                h_out = y_out_end - y_out
                w_out = x_out_end - x_out

                # Build faded masks
                ov_h_out = max(0, min(overlap_h, h_out - 1))
                ov_w_out = max(0, min(overlap_w, w_out - 1))
                
                weight_h = torch.ones((h_out,), device=decoded_tile.device, dtype=decoded_tile.dtype)
                weight_w = torch.ones((w_out,), device=decoded_tile.device, dtype=decoded_tile.dtype)

                # Apply fades only on interior edges using cached ramps (avoid fading on outer image borders)
                if ov_h_out > 0:
                    if y_lat > 0:  # Not top edge
                        weight_h[:ov_h_out] = ramp_cache['h'][:ov_h_out]
                    if y_lat_end < H:  # Not bottom edge
                        weight_h[-ov_h_out:] = 1 - ramp_cache['h'][:ov_h_out]
                if ov_w_out > 0:
                    if x_lat > 0:  # Not left edge
                        weight_w[:ov_w_out] = ramp_cache['w'][:ov_w_out]
                    if x_lat_end < W:  # Not right edge
                        weight_w[-ov_w_out:] = 1 - ramp_cache['w'][:ov_w_out]

                # Separable application (no 2D mask to save memory)
                weight_h_5d = weight_h.view(1, 1, 1, h_out, 1)
                weight_w_5d = weight_w.view(1, 1, 1, 1, w_out)
                decoded_tile.mul_(weight_h_5d).mul_(weight_w_5d)

                # Accumulate (move to result device if different)
                if result.device != decoded_tile.device:
                    decoded_tile = decoded_tile.to(result.device)
                    weight_h_5d = weight_h_5d.to(result.device)
                    weight_w_5d = weight_w_5d.to(result.device)
                
                result[:, :, : decoded_tile.shape[2], y_out:y_out_end, x_out:x_out_end] += decoded_tile
                count[:, :, :, y_out:y_out_end, x_out:x_out_end].addcmul_(weight_h_5d, weight_w_5d)

        # Move result back to inference device if needed and normalize
        if result.device != z.device:
            result = result.to(z.device)
            count = count.to(z.device)
        result.div_(count.clamp(min=1e-6)) # In-place normalize

        if z.shape[2] == 1:  # single frame
            result = result.squeeze(2)

        return result

    def forward(
        self, x: torch.FloatTensor, mode: Literal["encode", "decode", "all"] = "all", **kwargs
    ):
        # x: [b c t h w]
        if mode == "encode":
            h = self.encode(x)
            return h.latent_dist
        elif mode == "decode":
            h = self.decode(x)
            return h.sample
        else:
            h = self.encode(x)
            h = self.decode(h.latent_dist.mode())
            return h.sample

    def load_state_dict(self, state_dict, strict=False, assign=False):
        # Newer version of diffusers changed the model keys,
        # causing incompatibility with old checkpoints.
        # They provided a method for conversion.
        # We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class VideoAutoencoderKLWrapper(VideoAutoencoderKL):

    # 🚀 ADD THIS TO RESOLVE THE NO-SETTER ERROR
    @property
    def device(self):
        """
        Overrides the read-only property from the base class.
        Returns the device of the weights if they exist, 
        otherwise returns the manually set device.
        """
        try:
            return next(self.parameters()).device
        except (StopIteration, AttributeError):
            return getattr(self, "_manually_set_device", torch.device("cpu"))

    @device.setter
    def device(self, value):
        """
        Provides the 'setter' that Python is complaining is missing.
        This allows 'self.device = device' to work.
        """
        self._manually_set_device = value

    def __init__(
        self,
        *args,
        spatial_downsample_factor: int,
        temporal_downsample_factor: int,
        freeze_encoder: bool,
        **kwargs,
    ):
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.freeze_encoder = freeze_encoder
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        x = self.decode(z)
        return x

    def encode(self, x: torch.FloatTensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64)) -> torch.FloatTensor:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        p = super().encode(x, return_dict=return_dict, tiled=tiled, tile_size=tile_size,
                          tile_overlap=tile_overlap)
        # Use deterministic mode for tiled encoding to avoid artifacts
        z = p.mode().squeeze(2)
        return z

    def decode(self, z: torch.Tensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64)) -> torch.FloatTensor:
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = super().decode(z, return_dict=return_dict, tiled=tiled, tile_size=tile_size,
                          tile_overlap=tile_overlap).squeeze(2)
        return x

    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % 4 == 1
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: _memory_device_t,
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
            self.slicing_sample_min_size = split_size
            self.slicing_latent_min_size = max(1, split_size // self.temporal_downsample_factor)
        else:
            self.disable_slicing()
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))
                