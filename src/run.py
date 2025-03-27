import sys
import numpy as np

from arguments import parse_option
from ISO import query
from model import set_model
from loader import download_dataset, set_test_loader
from train_nn import train, test
from utils import set_reproductibility, Logger


def main():
    args = parse_option()
    sys.stdout = Logger(file_path=f'{args.output_dir}/log.txt')
    print(f'argument: {args}')

    set_reproductibility(seed=args.seed)

    print(f'dataset: {args.dataset}')
    X_train, y_train, X_val, y_val, X_test, y_test = download_dataset(dataset=args.dataset, 
                                                                      budget=args.budget)
        
    val_loader = set_test_loader(X=X_val, y=y_val, 
                                 image_size=args.image_size, 
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers)
    test_loader = set_test_loader(X=X_test, y=y_test, 
                                 image_size=args.image_size, 
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers)
    
    model = set_model(model_backbone=args.model_backbone, 
                      num_classes=args.num_classes, 
                      num_super_classes=args.num_super_classes).to(args.device)
    
    annotation_type = np.zeros(len(X_train), dtype=int)  # 0: unlabeled, 1: weak label, 2: full label
    
    for round in range(args.num_rounds):
        print('----------------------------------------------')
        print(f'Round: {round+1}/{args.num_rounds}')
        
        # ---- annotation ----
        print('query...')
        annotation_type = query(round=round,
                                budget=args.budget,
                                cost_weak=args.cost_weak,
                                X_train=X_train,
                                y_train=y_train,
                                annotation_type=annotation_type,
                                val_loader=val_loader,
                                model=model,
                                num_classes=args.num_classes, 
                                num_super_classes=args.num_super_classes,
                                optimizer=args.optimizer,
                                lr=args.lr,
                                image_size=args.image_size, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                num_epochs=args.num_epochs,
                                uncertainty=args.uncertainty,
                                device=args.device,)
        
        print('number of weak supervision: %d/%d (%.2f%%)'%(
            sum(annotation_type==1), len(X_train), 
            (sum(annotation_type==1)/len(X_train))*100))
        
        print('number of full supervision: %d/%d (%.2f%%)'%(
            sum(annotation_type==2), len(X_train), 
            (sum(annotation_type==2)/len(X_train))*100))

        # ---- training ----
        print('training...')
        
        model = model.init().to(args.device)
        model, _ = train(X_train=X_train,
                        y_train=y_train,
                        annotation_type=annotation_type,
                        val_loader=val_loader,
                        model=model,
                        num_classes=args.num_classes, 
                        num_super_classes=args.num_super_classes,
                        optimizer=args.optimizer,
                        lr=args.lr,
                        image_size=args.image_size,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        num_epochs=args.num_epochs,
                        device=args.device)
        
        # ---- test ----
        test_acc = test(test_loader=test_loader,
                        model=model,
                        device=args.device)
        print(f'test accuracy: {test_acc:.2f} %')
    
    print('----------------------------------------------')


if __name__ == '__main__':
    main()