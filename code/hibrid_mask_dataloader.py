import torch
import torch.nn as nn

from nvidia.dali.plugin.pytorch import DALIGenericIterator

class HybridMaskedVideoIterator:
    """
    Iterator híbrido para vídeos que combina:
    1. DALI: carregamento e pré-processamento de vídeos (GPU)
    2. PyTorch: aplicação de máscaras temporais/espaciais
    
    Args:
        root_dir: Diretório com vídeos
        input_shape: Tupla (altura, largura) para redimensionamento
        sequence_length: Número de frames por sequência
        batch_size: Tamanho do batch
        cover_factor: Fração dos frames/pixels a serem cobertos (0.0-1.0)
        square_size: Tamanho do quadrado da máscara (para máscaras espaciais)
        frame_coverage: Fração de frames a serem mascarados (para máscaras temporais)
        mask_type: Tipo de máscara ("spatial", "temporal", "spatiotemporal")
        cover_method: Método de cobertura ("ones", "zeros", "gray", "random")
        num_threads: Threads para DALI
        device_id: ID da GPU
        train: Modo de treinamento (habilita data augmentation)
        stride: Espaçamento entre frames
        step: Passo entre sequências
    """
    
    def __init__(
        self,
        root_dir: str,
        pipeline,
        input_shape: tuple,
        sequence_length: int,
        batch_size: int,
        mask_prob: float = 0.9,
        cover_factor: float = 0.3,
        square_size: int = 3,
        frame_coverage: float = 0.3,
        mask_type: str = "spatiotemporal",
        cover_method: str = "random",
        device_id: int = 0,
    ):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.input_height, self.input_width = input_shape
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.cover_factor = cover_factor
        self.square_size = square_size
        self.frame_coverage = frame_coverage
        self.mask_type = mask_type
        self.cover_method = cover_method
        self.device_id = device_id
        self.mask_prob = mask_prob
        
        # Calcular parâmetros de mascaramento
        self._calculate_mask_parameters()
        
        # Construir pipeline DALI para vídeos
        self.pipeline = pipeline
        
        # Build the pipeline
        self.pipeline.build()
        
        # Criar iterator DALI para PyTorch
        self.dali_iterator = DALIGenericIterator(
            [self.pipeline],
            output_map=['videos', 'labels'],
            auto_reset=True,
            reader_name='seq'
        )
    
    def _calculate_mask_parameters(self):
        """Calcula parâmetros para diferentes tipos de máscara."""
        
        # Para máscaras espaciais: número de quadrados por frame
        if self.mask_type in ["spatial", "spatiotemporal"]:
            total_pixels = self.input_height * self.input_width
            square_area = self.square_size * self.square_size
            self.n_squares = int(self.cover_factor * total_pixels / square_area)
            
            # Garantir pelo menos 1 quadrado se cover_factor > 0
            if self.n_squares == 0 and self.cover_factor > 0:
                self.n_squares = 1
        else:
            self.n_squares = 0
        
        # Para máscaras temporais: número de frames a mascarar
        if self.mask_type in ["temporal", "spatiotemporal"]:
            self.n_frames_to_mask = int(self.frame_coverage * self.sequence_length)
            
            # Garantir pelo menos 1 frame se frame_coverage > 0
            if self.n_frames_to_mask == 0 and self.frame_coverage > 0:
                self.n_frames_to_mask = 1
        else:
            self.n_frames_to_mask = 0
    
    def __iter__(self):
        """Retorna o iterator."""
        return self
    
    def __next__(self):
        """
        Retorna um batch (masked_videos, original_videos, labels).
        Aplicação da máscara é feita em PyTorch na GPU.
        """
        # Obter batch do DALI
        dali_output = next(self.dali_iterator)
        original_videos = dali_output[0]['videos']  # Formato: [B, C, F, H, W]
        labels = dali_output[0]['labels']  # Labels do vídeo
        
        # Clonar batch para versão mascarada
        masked_videos = original_videos.clone()
        
        # Aplicar máscaras usando PyTorch
        self._apply_video_masks_pytorch(masked_videos)
        
        return masked_videos, original_videos, labels
    
    def _apply_video_masks_pytorch(self, videos: torch.Tensor):
        """
        Aplica máscaras a sequências de vídeo usando PyTorch.
        
        Args:
            videos: Tensor de vídeos [B, C, F, H, W] no GPU
        """
        batch_size = videos.shape[0]
        n_channels = videos.shape[1]
        n_frames = videos.shape[2]
        
        for b in range(batch_size):
            # Decidir aleatoriamente se mascara este vídeo (50% chance)
            if torch.rand(1, device=videos.device).item() > self.mask_prob:
                continue
            
            # Aplicar máscara temporal (se configurado)
            if self.mask_type in ["temporal", "spatiotemporal"] and self.n_frames_to_mask > 0:
                self._apply_temporal_mask(videos[b], n_frames)
            
            # Aplicar máscara espacial (se configurado)
            if self.mask_type in ["spatial", "spatiotemporal"] and self.n_squares > 0:
                self._apply_spatial_mask(videos[b], n_frames, n_channels)
    
    def _apply_temporal_mask(self, video: torch.Tensor, n_frames: int):
        """
        Aplica máscara temporal (frames inteiros).
        
        Args:
            video: Tensor de um único vídeo [C, F, H, W]
            n_frames: Número de frames no vídeo
        """
        # Selecionar frames aleatórios para mascarar
        frames_to_mask = torch.randperm(n_frames, device=video.device)[:self.n_frames_to_mask]
        
        for frame_idx in frames_to_mask:
            # Determinar valor da máscara
            if self.cover_method == "ones":
                mask_value = 1.0
                video[:, frame_idx, :, :] = mask_value
            
            elif self.cover_method == "zeros":
                mask_value = 0.0
                video[:, frame_idx, :, :] = mask_value
            
            elif self.cover_method == "gray":
                mask_value = 0.5
                video[:, frame_idx, :, :] = mask_value
            
            elif self.cover_method == "random":
                # Valor aleatório para cada pixel em todos os canais
                mask_value = torch.rand(
                    video.shape[0],  # Canais
                    video.shape[2],  # Altura
                    video.shape[3],  # Largura
                    device=video.device
                )
                video[:, frame_idx, :, :] = mask_value
    
    def _apply_spatial_mask(self, video: torch.Tensor, n_frames: int, n_channels: int):
        """
        Aplica máscara espacial (quadrados em frames específicos).
        
        Args:
            video: Tensor de um único vídeo [C, F, H, W]
            n_frames: Número de frames no vídeo
            n_channels: Número de canais (geralmente 3)
        """
        # Para cada frame, decidir se aplica máscara espacial
        for f in range(n_frames):
            # Chance de aplicar máscara neste frame
            if torch.rand(1, device=video.device).item() > self.mask_prob:
                continue
            
            # Gerar posições aleatórias para os quadrados neste frame
            rows = torch.randint(
                0,
                self.input_height - self.square_size,
                (self.n_squares,),
                device=video.device
            )
            
            cols = torch.randint(
                0,
                self.input_width - self.square_size,
                (self.n_squares,),
                device=video.device
            )
            
            # Aplicar cada quadrado
            for i in range(self.n_squares):
                r, c = rows[i], cols[i]
                
                # Determinar valor da máscara
                if self.cover_method == "ones":
                    mask_value = 1.0
                    video[:, f, r:r+self.square_size, c:c+self.square_size] = mask_value
                
                elif self.cover_method == "zeros":
                    mask_value = 0.0
                    video[:, f, r:r+self.square_size, c:c+self.square_size] = mask_value
                
                elif self.cover_method == "gray":
                    mask_value = 0.5
                    video[:, f, r:r+self.square_size, c:c+self.square_size] = mask_value
                
                elif self.cover_method == "random":
                    # Valor aleatório para cada canal
                    mask_value = torch.rand(
                        n_channels,
                        self.square_size,
                        self.square_size,
                        device=video.device
                    )
                    video[:, f, r:r+self.square_size, c:c+self.square_size] = mask_value
    
    def reset(self):
        """Reseta o iterator DALI."""
        self.dali_iterator.reset()
    
    def __len__(self):
        return len(self.dali_iterator)
    
if __name__ == "__main__":
    from configParser import ConfigParser
    from base_dataloader import video_pipe
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    os.environ['DALI_DISABLE_NVML'] = '1'

    def view_masked_videos(masked_video, original_video, n_frames_to_show=10):
        masked_video_cpu = masked_video.cpu().numpy()
        original_video_cpu = original_video.cpu().numpy()
        n_batches = masked_video_cpu.shape[0]

        for b in range(n_batches):
            masked_video_np = masked_video_cpu[b]
            original_video_np = original_video_cpu[b]

            masked_frames = []
            original_frames = []
            
            for f in range(n_frames_to_show):
                masked_frame = masked_video_np[:, f, :, :].transpose(1, 2, 0)  # Para H, W, C
                original_frame = original_video_np[:, f, :, :].transpose(1, 2, 0)

                masked_frames.append(masked_frame)
                original_frames.append(original_frame)
                
            masked_horizontal = np.concatenate(masked_frames, axis=1)
            original_horizontal = np.concatenate(original_frames, axis=1)
            
            # Criar figura com duas linhas
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(n_frames_to_show * 3, 6))
            
            # Mostrar sequência mascarada
            ax1.imshow(masked_horizontal)
            ax1.set_title(f'Batch {b+1} - Mascarado ({n_frames_to_show} frames)')
            ax1.axis('off')
            
            # Mostrar sequência original
            ax2.imshow(original_horizontal)
            ax2.set_title(f'Batch {b+1} - Original ({n_frames_to_show} frames)')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    configFile = "config.json"
    config = ConfigParser(configFile).config
    
    shape = (config['input_size'][0], config['input_size'][1])
    trainPipe = video_pipe(file_root=config['trainPath'], train=True, shape=shape,
                        sequence_length=config['seqLen'], stride=config['fstride'], step=config['cstride'],
                        batch_size=config['batch_size'])
    trainPipe.build()

    hybridIter = HybridMaskedVideoIterator(root_dir=config['trainPath'],
                                       pipeline=trainPipe,
                                       input_shape=shape,
                                       sequence_length=config['seqLen'],
                                       batch_size=config['batch_size'],
                                       square_size=10,
                                       mask_type= "spatial",
                                       mask_prob=0.95)
    
    tloader = tqdm(hybridIter,unit='batch')

    for data in tloader:
        masked_videos, original_videos,_ = data
        print(masked_videos.shape)
        print(original_videos.shape)
        
        view_masked_videos(masked_videos, original_videos, n_frames_to_show=10)
        break
