from Classes.picturedrawer import PictureDrawer

drawer = PictureDrawer(5, hexagon_status=True, random_status=False)
drawer.generate_images(100, 'first_stage', update=False)

