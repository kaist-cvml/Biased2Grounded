import math

ext_dic = {
    'left': lambda x1, y1, x2, y2: x1 < x2,  
    'right': lambda x1, y1, x2, y2: x1 > x2}

n_shot_support = [{'image': '/projects/exp/fruits/2/Peach8_banana35_tb.jpg', 'question': 'a peach on top of a banana', 'answer': 'Yes'},
                  {'image': '/projects/exp/fruits/2/Peach8_banana35_tb.jpg', 'question': 'a banana on top of a peach', 'answer': 'No'},
                  {'image': '/projects/exp/fruits/2/Peach8_banana35_tb.jpg', 'question': 'a peach on the bottom of a banana', 'answer': 'No'},
                  {'image': '/projects/exp/fruits/2/Peach8_banana35_tb.jpg', 'question': 'a banana on the bottom of a peach', 'answer': 'Yes'},
                  
                  {'image': '/projects/exp/fruits/2/Peach8_banana35_lr.jpg', 'question': 'a peach on the left of a banana', 'answer': 'Yes'},
                  {'image': '/projects/exp/fruits/2/Peach8_banana35_lr.jpg', 'question': 'a banana on the left of peach', 'answer': 'No'},
                  {'image': '/projects/exp/fruits/2/Peach8_banana35_lr.jpg', 'question': 'a peach on the right of a banana', 'answer': 'No'},
                  {'image': '/projects/exp/fruits/2/Peach8_banana35_lr.jpg', 'question': 'a banana on the right of a peach', 'answer': 'Yes'}]

pos2idx = {'top':0,
           'above':0,
           'over':0,
           'bottom':1,
           'underneath':1,
           'below':1,
           'left':2,
           'west':2,
           'right':3,
           'east': 3}

train_coor_dic = {'above_below': lambda c1, c2: [f'a {c1} on the top of a {c2}',
                                                 f'a {c2} on the bottom of a {c1}'],                                         

                  'left_right': lambda c1, c2: [f'a {c1} on the left of a {c2}',
                                                f'a {c2} on the right of a {c1}']}

epsilon = 1e-10
radius = 2

coor_dic = {'above': lambda x1, y1, x2, y2: y2-y1,
            'top': lambda x1, y1, x2, y2: y2-y1,
            'over': lambda x1, y1, x2, y2: y2-y1,
            'upper': lambda x1, y1, x2, y2: y2-y1,
                        
            'below': lambda x1, y1, x2, y2: y1-y2,
            'low': lambda x1, y1, x2, y2: y1-y2,
            'lower': lambda x1, y1, x2, y2: y1-y2,
            'bottom': lambda x1, y1, x2, y2: y1-y2,
            'underneath': lambda x1, y1, x2, y2: y1-y2,
            'beneath': lambda x1, y1, x2, y2: y1-y2,
            'under': lambda x1, y1, x2, y2: y1-y2,
            'blocking': lambda x1, y1, x2, y2: (y1-y2)/2,
            
            'left': lambda x1, y1, x2, y2: x2-x1,
            'right': lambda x1, y1, x2, y2: x1-x2,

            'front': lambda x1, y1, x2, y2: y1-y2,
            'behind': lambda x1, y1, x2, y2: y2-y1,

            'between': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),
                        
            # 'side': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),
            'near': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),
            'next': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),
            'close': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),
            'connected': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),
            'touching': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),

            'outside': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),
            'inside': lambda x1, y1, x2, y2: 1/(math.hypot(x1 - x2, y2 - y1)+epsilon),

            'opposite': lambda x1, y1, x2, y2: math.hypot(x1 - x2, y2 - y1),
            'distant': lambda x1, y1, x2, y2: math.hypot(x1 - x2, y2 - y1),
            'away': lambda x1, y1, x2, y2: math.hypot(x1 - x2, y2 - y1),
            'far': lambda x1, y1, x2, y2: math.hypot(x1 - x2, y2 - y1),
            'farther': lambda x1, y1, x2, y2: math.hypot(x1 - x2, y2 - y1)}

coor_dic_one = {'top': lambda x, y, delta_x, delta_y: delta_y-y,
                'upper': lambda x, y, delta_x, delta_y: delta_y-y,
            
                'bottom': lambda x, y, delta_x, delta_y: y-delta_y,       
                'lower': lambda x, y, delta_x, delta_y: y-delta_y,
                
                'left': lambda x, y, delta_x, delta_y: delta_x-x,
                'left-hand': lambda x, y, delta_x, delta_y: delta_x-x,
                'right': lambda x, y, delta_x, delta_y: x-delta_x,
                'right-hand': lambda x, y, delta_x, delta_y: x-delta_x,
                
                'behind': lambda x, y, delta_x, delta_y: delta_y-y,
                'centrally': lambda x, y, delta_x, delta_y: math.hypot(x - delta_x, y - delta_y),
                'central': lambda x, y, delta_x, delta_y: math.hypot(x - delta_x, y - delta_y),
                'center': lambda x, y, delta_x, delta_y: math.hypot(x - delta_x, y - delta_y),
                'middle': lambda x, y, delta_x, delta_y: math.hypot(x - delta_x, y - delta_y),
                
                'top-left': lambda x, y, delta_x, delta_y: (delta_y-y)*(delta_x-x),
                'upper-left': lambda x, y, delta_x, delta_y: (delta_y-y)*(delta_x-x),
                'top-right': lambda x, y, delta_x, delta_y: (delta_y-y)*(x-delta_x),
                'upper-right': lambda x, y, delta_x, delta_y: (delta_y-y)*(x-delta_x),
                
                'lower-left': lambda x, y, delta_x, delta_y: (y-delta_y)*(delta_x-x),
                'lower-right': lambda x, y, delta_x, delta_y: (y-delta_y)*(x-delta_x),
                'bottom-left': lambda x, y, delta_x, delta_y: (y-delta_y)*(delta_x-x),
                'bottom-right': lambda x, y, delta_x, delta_y: (y-delta_y)*(x-delta_x)}

lr_templates = [
    lambda c1, c2: f'{c1} is left to the {c2}.',
    lambda c1, c2: f'{c2} is left to the {c1}.',
    lambda c1, c2: f'{c2} is right to the {c1}.',
    lambda c1, c2: f'{c1} is right to the {c2}.',

    lambda c1, c2: f'The position of {c1} is left to the {c2}.',
    lambda c1, c2: f'The position of {c2} is left to the {c1}.',
    lambda c1, c2: f'The position of {c2} is right to the {c1}.',
    lambda c1, c2: f'The position of {c1} is right to the {c2}.',

    lambda c1, c2: f'{c1} is positioned to the left of {c2}.',
    lambda c1, c2: f'{c2} is positioned to the left of {c1}.',
    lambda c1, c2: f'{c2} is positioned to the right of {c1}.',
    lambda c1, c2: f'{c1} is positioned to the right of {c2}.',

    lambda c1, c2: f'{c1} appears to the left of {c2}.',
    lambda c1, c2: f'{c2} appears to the left of {c1}.',
    lambda c1, c2: f'{c2} appears to the right of {c1}.',
    lambda c1, c2: f'{c1} appears to the right of {c2}.',

    lambda c1, c2: f'{c1} is located to the left of {c2}.',
    lambda c1, c2: f'{c2} is located to the left of {c1}.',
    lambda c1, c2: f'{c2} is located to the right of {c1}.',
    lambda c1, c2: f'{c1} is located to the right of {c2}.',

    lambda c1, c2: f'{c1} is placed on the left side of {c2}.',
    lambda c1, c2: f'{c2} is placed on the left side of {c1}.',
    lambda c1, c2: f'{c2} is placed on the right side of {c1}.',
    lambda c1, c2: f'{c1} is placed on the right side of {c2}.',

    lambda c1, c2: f'{c1} is aligned to the left of {c2}.',
    lambda c1, c2: f'{c2} is aligned to the left of {c1}.',
    lambda c1, c2: f'{c2} is aligned to the right of {c1}.',
    lambda c1, c2: f'{c1} is aligned to the right of {c2}.',
]

tb_templates = [
    lambda c1, c2: f'{c1} is above {c2}.',
    lambda c1, c2: f'{c2} is above {c1}.',
    lambda c1, c2: f'{c2} is below {c1}.',
    lambda c1, c2: f'{c1} is below {c2}.',

    lambda c1, c2: f'{c1} is on top of {c2}.',
    lambda c1, c2: f'{c2} is on top of {c1}.',
    lambda c1, c2: f'{c2} is underneath {c1}.',
    lambda c1, c2: f'{c1} is underneath {c2}.',

    lambda c1, c2: f'{c1} is over {c2}.',
    lambda c1, c2: f'{c2} is over {c1}.',
    lambda c1, c2: f'{c2} is under {c1}.',
    lambda c1, c2: f'{c1} is under {c2}.',

    lambda c1, c2: f'{c1} is positioned above {c2}.',
    lambda c1, c2: f'{c2} is positioned above {c1}.',
    lambda c1, c2: f'{c2} is positioned below {c1}.',
    lambda c1, c2: f'{c1} is positioned below {c2}.',

    lambda c1, c2: f'{c1} is situated above {c2}.',
    lambda c1, c2: f'{c2} is situated above {c1}.',
    lambda c1, c2: f'{c2} is situated below {c1}.',
    lambda c1, c2: f'{c1} is situated below {c2}.',

    lambda c1, c2: f'{c1} is located on top of {c2}.',
    lambda c1, c2: f'{c2} is located on top of {c1}.',
    lambda c1, c2: f'{c2} is located on the bottom of {c1}.',
    lambda c1, c2: f'{c1} is located on the bottom of {c2}.',
]

revision_templates = {'right':'left',
                      'left':'right',
                      'above':'below',
                      'below':'above',
                      'top':'bottom',
                      'bottom':'top',
                      'underneath':'over',
                      'over':'underneath',
                      'near':'distant',
                      'next':'far away', 
                      'side':'opposite',
                      'front':'behind', 
                      'behind': 'in front of'}