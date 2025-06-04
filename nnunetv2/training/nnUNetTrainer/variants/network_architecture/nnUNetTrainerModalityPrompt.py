from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network_architecture.modality_prompt_wrapper import ModalityPromptWrapper


class nnUNetTrainerModalityPrompt(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        base_net = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return ModalityPromptWrapper(base_net, num_input_channels)

