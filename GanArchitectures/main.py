from factory import ModelFactory
from programparser import get_parser

if __name__ == "__main__":
    facto = ModelFactory() # Get factory
    parser = get_parser()
    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    resume_training = args.resume_training
    model, visiondata = facto.get_trainable_model(model_name, dataset, batch_size)
    if model_name == "WGANCP" or model_name == "WGANGP":
        epochs = args.generator_iter
    model.train(visiondata, epochs, resume_training=resume_training)