from utils.datarepository import DataRepository
from models.gans import MLPGan
from models.gans import DCGan

repo = DataRepository.get_instance()
data = repo.get_data("MNIST", 64, 1)
Gan = DCGan(1)
Gan.train(data, 100)