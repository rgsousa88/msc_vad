from models import *

class ModelFactory:
    """Factory para criar instâncias de modelos baseado em uma string de entrada."""
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> nn.Module:
        """
        Cria uma instância do modelo especificado.
        
        Args:
            model_name: Nome do modelo a ser criado.
            **kwargs: Argumentos para o construtor do modelo.
            
        Returns:
            Instância do modelo.
            
        Raises:
            ValueError: Se o nome do modelo não for reconhecido.
        """
        model_registry = {
            'ConvBlock3D': ConvBlock3D,
            'UpConvBlock3D': UpConvBlock3D,
            'CNN3D': CNN3D,
            'CNN3DRes': CNN3DRes,
            'CameraClassifier': CameraClassifier,
            'CNN3DRecon': CNN3DRecon,
            'CNN3DResRecon': CNN3DResRecon,
            'CNN3DResReconSkipV1': CNN3DResReconSkipV1,
            'CNN3DResReconSkipV2': CNN3DResReconSkipV2,
            'SSMTLModel': SSMTLModel,
            'SSMTLAutoencoder': SSMTLAutoEncoder,
            'SSTMLAutoEncArrow':SSMTLAutoEncArrow
        }
        
        if model_name not in model_registry:
            available_models = ', '.join(sorted(model_registry.keys()))
            raise ValueError(
                f"Modelo '{model_name}' não reconhecido. "
                f"Modelos disponíveis: {available_models}"
            )
        
        model_class = model_registry[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Retorna uma lista com os nomes dos modelos disponíveis."""
        return list(cls.create_model.__annotations__.get('model_name', '').split('|'))


# Exemplo de uso:
if __name__ == "__main__":
    factory = ModelFactory()
    
    # Criando diferentes modelos
    conv_block = factory.create_model('ConvBlock3D', in_channel=16, out_channel=32)
    print(f"ConvBlock3D criado: {conv_block}")
    
    cnn3d = factory.create_model('CNN3D', in_channel=16, out_channel=32, temp_pool=1)
    print(f"CNN3D criado: {cnn3d}")
    
    classifier = factory.create_model('CameraClassifier', n_classes=13, temp_pool=5)
    print(f"CameraClassifier criado: {classifier}")
    
    recon = factory.create_model('CNN3DRecon', in_channel=32, out_channel=64)
    print(f"CNN3DRecon criado: {recon}")
    
    # Listando modelos disponíveis
    print(f"\nModelos disponíveis: {factory.get_available_models()}")
    
    # Exemplo com erro
    try:
        invalid_model = factory.create_model('ModeloInexistente')
    except ValueError as e:
        print(f"\nErro esperado: {e}")