import sys
import os
import importlib
import models

import torch.nn as nn

importlib.reload(models)

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "models/IR"))  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "models/MGMN"))

class MGMN(nn.Module):
    def __init__(self, input_feature_dim=0, args=None):
        super().__init__()
        self.args = args

        # --------- LANGUAGE ENCODING ---------
        # module = importlib.import_module(args.language_module)
        # self.lang = module.LangModule(args)
        module = importlib.import_module(args.parse_module)
        self.lang = module.ParseModule(args)


        # --------- INSTANCE ENCODING ---------
        if args.attribute_module:
            module = importlib.import_module(args.attribute_module)
            self.attribute = module.AttributeModule(input_feature_dim, args)

        if args.textguided_module:
            module = importlib.import_module(args.textguided_module)
            self.textguided = module.TextGuidedModule(input_feature_dim, args)

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        ### language module
        data_dict = self.lang(data_dict)

        ### attribute module
        if self.args.attribute_module:
            data_dict = self.attribute(data_dict)

        ### relation module
        if self.args.textguided_module:
            data_dict = self.textguided(data_dict)

        return data_dict
