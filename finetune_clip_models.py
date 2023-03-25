import torch
import torch.nn as nn

"""
All the CLIP finetuning models are defined in this file.
While training, these models are used to train our text and image embeddings.
These models take text and image embeddings generated from CLIP and outputs 
the new embeddings when run through these networks.

"""
class CLIP_1(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 512
    ):
        super().__init__()

        self.fc_image = nn.Linear(input_size, output_size)
        self.fc_text = nn.Linear(input_size, output_size)
        
        self.gelu_image = nn.GELU()
        self.gelu_text = nn.GELU()
        
    def forward(self, image_features, text_features):
        text_embedding = self.fc_text(text_features)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding  = torch.nn.functional.normalize(text_embedding, dim=-1)

        image_embedding = self.fc_image(image_features)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)


        return text_embedding, image_embedding
    
class CLIP_2(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 512
    ):
        super().__init__()


        self.fc_text = nn.Linear(input_size, output_size)
        self.gelu_text = nn.GELU()
        
        self.fc_image = nn.Linear(input_size, output_size)
        self.gelu_image = nn.GELU()
        
    def forward(self, image_features, text_features):
        text_embedding = self.fc_text(text_features)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding = self.fc_text(text_embedding)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding  = torch.nn.functional.normalize(text_embedding, dim=-1)

        image_embedding = self.fc_image(image_features)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = self.fc_image(image_embedding)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)



        return text_embedding, image_embedding
    
class CLIP_3(nn.Module):
    def __init__(
        self,
        input_size = 512,
        hidden_size = 512,
        output_size = 512
    ):
        super().__init__()

        self.lstm_text = nn.LSTM(input_size, hidden_size)
        self.lstm_image = nn.LSTM(input_size, hidden_size)
        self.fc_text = nn.Linear(hidden_size, output_size)
        self.fc_image = nn.Linear(hidden_size, output_size)
        self.gelu_text = nn.GELU()
        self.gelu_image = nn.GELU()
        
    def forward(self, image_features, text_features):

        text_embedding = self.lstm_text(text_features)[0]
        text_embedding = self.fc_text(text_embedding)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding  = torch.nn.functional.normalize(text_embedding, dim=-1)


        
        image_embedding = self.lstm_image(image_features)[0]
        image_embedding = self.fc_image(image_embedding)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)


        return text_embedding, image_embedding