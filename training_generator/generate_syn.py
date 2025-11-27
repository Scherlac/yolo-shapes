# generate synthetic svg data with rectangles and circles for mmyolo training

import random
import numpy as np
import os
import json

# image magic for svg to png conversion
# from PIL import Image
#from cairosvg import svg2png

# svg generation
# from svgwrite import Drawing, Group, Rect, Circle, Ellipse, Polygon
import drawsvg 

# % pip install numpy Pillow cairosvg svgwrite

def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

def ellipse_radius(angle, a, b):
    return a * b / np.sqrt((b * np.cos(angle)) ** 2 + (a * np.sin(angle)) ** 2) 

def shape_overlap(shape1, shape2):

    dx = shape2['x'] - shape1['x']
    dy = shape2['y'] - shape1['y']

    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    
    ext1 = ellipse_radius(angle-shape1['rot'], shape1['w']/2, shape1['h']/2)
    ext2 = ellipse_radius(angle-shape2['rot'], shape2['w']/2, shape2['h']/2)
    # ext1 = np.sqrt(shape1['w'] ** 2 + shape1['h'] ** 2) * 0.5
    # ext2 = np.sqrt(shape2['w'] ** 2 + shape2['h'] ** 2) * 0.5

    # minimum of ext1 and ext2
    min = np.minimum(ext1, ext2)

    overlap = dist - (ext1 + ext2 - 0.2 * min)

    if overlap > 0:
        return 0, ext1, ext2, dist
    
    return 1, ext1, ext2, dist

def overlap_check(shapes, shape, overlap_threshold=0.2):
        for s in shapes:
            overlap_factor, ext1, ext2, dist = shape_overlap(s, shape)
            if overlap_factor > overlap_threshold:
                
                return False
    
        return True

def border_check(shape, width, height, border=0.02):
    if (shape['x'] - shape['w']/2 <  border * width) or shape['x'] + shape['w']/2 > (1 - border) * width:
        return False
    if (shape['y'] - shape['h']/2 <  border * height) or shape['y'] + shape['h']/2 > (1 - border) * height:
        return False
    
    return True

def shape_check(shapes, shape, width, height, border=0.02):

    if not border_check(shape, width, height, border):
        return False
    
    if not overlap_check(shapes, shape):
        return False

    return True    

def generate_data(num_images=100, # number of images to generate
                  width=640, # image width
                  height=640, # image height
                  shape_types=['circle', 'rect', 'ellipsis'], # types of shapes to generate
                  min_size=30, # min size of objects
                  max_size=130, # max size of objects
                  min_objects=10, # min number of objects
                  max_objects=20, # max number of objects
                  overlap=0.8, # overlap threshold in percent
                  ):

    """
    Generate synthetic data for mmyolo training
    """

    data = []
    random.seed(32)

    for i in range(num_images):
        shapes = []
        num_objects = random.randint(min_objects, max_objects)
        background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for j in range(num_objects):
            shape_type = random.choice(shape_types)

            if shape_type == 'circle':
                w = h = random.randint(min_size, max_size)
            else:
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)

            while True:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if color_distance(color, background_color) > 100:
                    break

            shape = {'type': shape_type, 'w': w, 'h': h, 'color': color}
            trys = 0
            while True:
                shape['x'] = random.randint(0, width)
                shape['y'] = random.randint(0, height)
                shape['rot'] = random.uniform(0, np.pi/2)

                if shape_check(shapes, shape, width, height):
                    break
                trys += 1

                if trys >= 100:
                    break
            
            if trys < 100:
                shapes.append(shape)

        image = {'width': width, 'height': height, 'background_color': background_color, 'shapes': shapes}
        
        data.append(image)
    return data

def generate_images(data, svg_output_dir='data/svg', png_output_dir='data/png'):
    """
    Generate svg files
    """

    if not os.path.exists(svg_output_dir):
        os.makedirs(svg_output_dir)

    if not os.path.exists(png_output_dir):
        os.makedirs(png_output_dir)

    for i, image in enumerate(data):
        # Src: https://pypi.org/project/drawsvg/
        dwg = drawsvg.Drawing(
            height=image['height'],
            width=image['width']
        )
            
        
        # background
        dwg.append(drawsvg.Rectangle(
            x=0, y=0, width=image['width'], height=image['height'],
            fill=f'rgb{image["background_color"]}'))

        for shape in image['shapes']:
            group = drawsvg.Group(
                transform=f'translate({shape["x"]}, {shape["y"]}) rotate({np.degrees(shape["rot"])})'
            )

            if shape['type'] == 'circle':
                group.append(drawsvg.Circle(
                    cx=0, cy=0,
                    r=shape['w']/2,
                    fill=f'rgb{shape["color"]}'
                    )
                )
            elif shape['type'] == 'rect':
                group.append(drawsvg.Rectangle(
                    x=-shape['w']/2, y=-shape['h']/2, 
                    width=shape['w'], height=shape['h'], 
                    fill=f'rgb{shape["color"]}'))
                
            elif shape['type'] == 'ellipsis':
                group.append(drawsvg.Ellipse(
                    cx=0, cy=0,
                    rx=shape['w']/2, ry=shape['h']/2,
                    fill=f'rgb{shape["color"]}'))
                    
            # elif shape['type'] == 'triangle':
            #     group.append(drawsvg.Lines(
            #         0, shape['h']/2,
            #         -shape['w']/2, shape['h']/2,
            #         shape['w']/2, shape['h']/2,
            #         fill=f'rgb{shape["color"]}',
            #         close=True
            #         ))

            dwg.append(group)

        dwg.save_svg(f'{svg_output_dir}/image_{i:04}.svg')
        dwg.save_png(f'{png_output_dir}/image_{i:04}.png')

        image['svg'] = f'{svg_output_dir}/image_{i:04}.svg'
        image['png'] = f'{png_output_dir}/image_{i:04}.png'



if __name__ == '__main__':
    data = generate_data()

    generate_images(data)

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)


                 
                 
