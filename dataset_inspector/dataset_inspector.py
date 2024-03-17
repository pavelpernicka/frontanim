#!/usr/bin/env python3
import gi
import os
from gi.repository import Gtk, GdkPixbuf, Gio
from PIL import Image
import io
import numpy as np
import random
import shutil

gi.require_version('Gtk', '3.0')

class ProgressDialog(Gtk.Dialog):
    def __init__(self, parent):
        Gtk.Dialog.__init__(self, "Saving Dataset", parent, 0)

        # Set dialog size
        self.set_default_size(300, 60)

        # Create progress bar
        self.progressbar = Gtk.ProgressBar()
        self.progressbar.set_size_request(280, 30)
        box = self.get_content_area()
        box.add(self.progressbar)

        # Connect signal to update progress
        self.connect("delete-event", Gtk.main_quit)
        self.show_all()

    def update_progress(self, fraction):
        self.progressbar.set_fraction(fraction)
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
            
class TestDatasetDialog(Gtk.Dialog):
    def __init__(self, parent):
        Gtk.Dialog.__init__(self, "Test Dataset", parent)

        self.set_default_size(200, 100)

        self.icon_theme = Gtk.IconTheme.get_default()

        ok_icon = self.icon_theme.load_icon("gtk-ok", 16, 0)
        ok_button = Gtk.Button.new_with_label("OK")
        ok_button.set_image(Gtk.Image.new_from_pixbuf(ok_icon))
        ok_button.connect("clicked", self.on_ok_clicked)

        cancel_icon = self.icon_theme.load_icon("gtk-cancel", 16, 0)
        cancel_button = Gtk.Button.new_with_label("No test dataset")
        cancel_button.set_image(Gtk.Image.new_from_pixbuf(cancel_icon))
        cancel_button.connect("clicked", self.on_cancel_clicked)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.get_content_area().add(vbox)

        label = Gtk.Label(label="Enter the percentage of the test dataset:")
        vbox.pack_start(label, False, False, 0)

        self.entry = Gtk.Entry()
        self.entry.set_placeholder_text("Percentage")
        self.entry.set_max_length(3)
        self.entry.connect("changed", self.on_entry_changed)
        vbox.pack_start(self.entry, False, False, 0)

        self.set_default_percentage(10)

        self.get_action_area().pack_end(cancel_button, False, False, 0)
        self.get_action_area().pack_end(ok_button, False, False, 0)

        self.show_all()

    def set_default_percentage(self, percentage):
        self.entry.set_text(str(percentage))

    def get_percentage(self):
        return self.entry.get_text()

    def on_entry_changed(self, entry):
        text = entry.get_text()
        try:
            value = int(text)
            if value < 0 or value > 100:
                entry.set_text("")  # Clear the entry if value is out of range
        except ValueError:
            entry.set_text("")  # Clear the entry if it's not a valid integer

    def on_ok_clicked(self, button):
        self.response(Gtk.ResponseType.OK)

    def on_cancel_clicked(self, button):
        self.response(Gtk.ResponseType.CANCEL)
        
class SimpleApp(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Frontanim dataset inspector")

        # Create a grid to organize widgets
        self.grid = Gtk.Grid()
        self.add(self.grid)

        # Create the menubar
        menubar = Gtk.MenuBar()
        self.grid.attach(menubar, 0, 0, 1, 1)

        # Create the Dataset menu
        dataset_menu = Gtk.Menu()
        dataset_menu_item = Gtk.MenuItem(label="Dataset")
        dataset_menu_item.set_submenu(dataset_menu)
        menubar.append(dataset_menu_item)

        # Add Open option to the Dataset menu
        open_menu_item = Gtk.ImageMenuItem.new_with_label("Open")
        open_menu_item.connect("activate", self.on_open_activate)
        open_image = Gtk.Image.new_from_icon_name("document-open", Gtk.IconSize.MENU)
        open_menu_item.set_image(open_image)
        dataset_menu.append(open_menu_item)

        # Add Save option to the Dataset menu
        self.save_menu_item = Gtk.ImageMenuItem.new_with_label("Save")
        self.save_menu_item.connect("activate", self.on_save_activate)
        save_image = Gtk.Image.new_from_icon_name("document-save", Gtk.IconSize.MENU)
        self.save_menu_item.set_image(save_image)
        dataset_menu.append(self.save_menu_item)

        # Add a separator
        dataset_menu.append(Gtk.SeparatorMenuItem())

        # Create the View menu
        view_menu = Gtk.Menu()
        view_menu_item = Gtk.MenuItem(label="View")
        view_menu_item.set_submenu(view_menu)
        menubar.append(view_menu_item)

        # Add "Show fronts" item to the View menu
        show_fronts_item = Gtk.CheckMenuItem(label="Show fronts")
        show_fronts_item.set_active(True)  # Checked by default
        show_fronts_item.connect("toggled", self.on_show_fronts_toggled)
        view_menu.append(show_fronts_item)
        
        # Add "Show model data" item to the View menu
        show_model_item = Gtk.CheckMenuItem(label="Show model data")
        show_model_item.set_active(True)  # Checked by default
        show_model_item.connect("toggled", self.on_show_model_toggled)
        view_menu.append(show_model_item)
        
        # Add "Split view" item to the View menu
        show_split_item = Gtk.CheckMenuItem(label="Split view")
        show_split_item.set_active(False)
        show_split_item.connect("toggled", self.on_split_toggled)
        view_menu.append(show_split_item)
        
        # Create the Help menu
        help_menu = Gtk.Menu()
        help_menu_item = Gtk.MenuItem(label="Help")
        help_menu_item.set_submenu(help_menu)
        menubar.append(help_menu_item)

        # Add "About" item to the Help menu
        about_menu_item = Gtk.MenuItem(label="About")
        about_menu_item.connect("activate", self.on_about_activate)
        help_menu.append(about_menu_item)

        # Create the sidebar
        self.sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.grid.attach(self.sidebar, 0, 1, 1, 2)

        # Add widgets to the sidebar
        self.treeview_label = Gtk.Label(label="")
        self.sidebar.pack_start(self.treeview_label, False, True, 0)

        self.treeview = Gtk.TreeView()
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.add(self.treeview)
        self.sidebar.pack_start(scrolled_window, True, True, 0)

        # Create the ListStore for the TreeView
        self.liststore = Gtk.ListStore(str, bool)
        self.treeview.set_model(self.liststore)

        # Add columns to the TreeView
        renderer_text = Gtk.CellRendererText()
        column_name = Gtk.TreeViewColumn("Name", renderer_text, text=0)
        self.treeview.append_column(column_name)

        renderer_toggle = Gtk.CellRendererToggle()
        renderer_toggle.connect("toggled", self.on_toggle_toggled)
        column_include = Gtk.TreeViewColumn("Include", renderer_toggle, active=1)
        self.treeview.append_column(column_include)

        # Create a group for the image and label
        self.image_group = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.grid.attach_next_to(self.image_group, self.sidebar, Gtk.PositionType.RIGHT, 2, 1)

        # Add widgets to the image group
        self.image_label = Gtk.Label(label="Please select a dataset to inspect")
        self.image_group.pack_start(self.image_label, False, True, 0)

        self.image_alignment = Gtk.Alignment.new(0.5, 0.5, 0, 0)
        self.image_group.pack_start(self.image_alignment, True, True, 0)

        # Create the action bar
        self.action_bar = Gtk.ActionBar()
        self.grid.attach_next_to(self.action_bar, self.sidebar, Gtk.PositionType.BOTTOM, 3, 1)

        # Add buttons to the action bar
        button_use = Gtk.Button(label="Use")
        button_use.set_image(Gtk.Image.new_from_icon_name("gtk-apply", Gtk.IconSize.BUTTON))
        button_use.connect("clicked", self.on_use_clicked)
        self.action_bar.pack_end(button_use)

        button_discard = Gtk.Button(label="Exclude")
        button_discard.set_image(Gtk.Image.new_from_icon_name("gtk-cancel", Gtk.IconSize.BUTTON))
        button_discard.connect("clicked", self.on_discard_clicked)
        self.action_bar.pack_end(button_discard)

        self.set_resizable(False)

        self.start()

    def start(self):
        self.opened_path = None
        self.display_fronts = True
        self.display_data = True
        self.display_split = False
        self.load_image(None)
        self.treeview.connect("cursor-changed", self.selected)
        self.save_menu_item.set_sensitive(False)

    def load_image(self, image):
        if not image:
            placeholder_pixbuf = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, True, 8, 512, 512)
            #placeholder_pixbuf.fill(0xFFFFFF00) # Fill with transparency
            placeholder_text = "Please select a dataset to inspect"
            self.image = Gtk.Image.new_from_pixbuf(placeholder_pixbuf)
            self.image_label.set_text(placeholder_text)
            self.image_alignment.add(self.image)
        else:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            # BytesIO to GdkPixbuf
            img_bytes.seek(0)
            stream = Gio.MemoryInputStream.new_from_data(img_bytes.read(), None)
            img_pixbuf = GdkPixbuf.Pixbuf.new_from_stream_at_scale(stream, 512, 512, True)
            if hasattr(self, 'image') and self.image is not None:
                self.image.set_from_pixbuf(img_pixbuf)
            else:
                self.image = Gtk.Image.new_from_pixbuf(img_pixbuf)
                self.image_alignment.add(self.image)

    def change_current_item(self, value):
        selection = self.treeview.get_selection()
        model, iter = selection.get_selected()
        if iter is not None:
            model[iter][1] = value

    def move_to_next_item(self):
        selection = self.treeview.get_selection()
        current_model, current_iter = selection.get_selected()

        if current_iter is None:
            current_iter = self.liststore.get_iter_first()
            if current_iter is None:
                return
        next_iter = self.liststore.iter_next(current_iter)

        if next_iter is not None:
            selection.select_iter(next_iter)
            self.treeview.scroll_to_cell(self.liststore.get_path(next_iter), None, True, 0.5, 0.5)
            self.selected()

    def on_use_clicked(self, button):
        self.change_current_item(True)
        self.move_to_next_item()

    def on_discard_clicked(self, button):
        self.change_current_item(False)
        self.move_to_next_item()

    def extract_date(self, filename):
        try:
            parts = os.path.splitext(filename)[0].split("_")
            if len(parts) != 3:
                raise ValueError("Invalid filename format")
            return int(parts[2]), int(parts[1]), int(parts[0])  # Year, Month, Day
        except (ValueError, IndexError):
            return None

    def on_open_activate(self, menu_item):
        dialog = Gtk.FileChooserDialog(
            title="Choose a folder",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
            buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "Open", Gtk.ResponseType.OK)
        )

        dialog.set_default_size(800, 400)
        dialog.set_current_folder(os.path.expanduser("~"))

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            folder_path = dialog.get_filename()
            print("Selected folder:", folder_path)
            file_list=os.listdir(folder_path)
            if ("data" in file_list) and ("fronts" in file_list):
                self.opened_path = folder_path
                image_list=os.listdir(folder_path + "/data")
                sorted_files = sorted((f for f in image_list if self.extract_date(f) is not None), key=self.extract_date)
                if(len(sorted_files)>0):
                    self.save_menu_item.set_sensitive(True)
                    self.liststore.clear()
                    for file in sorted_files:
                        if file.endswith(".png"):
                            self.liststore.append([file, True])
                    first_iter = self.liststore.get_iter_first()
                    self.treeview.get_selection().select_iter(first_iter)
                    self.selected()
            else:
                dialog.destroy()
                error_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    flags=0,
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK,
                    text="Error",
                )
                error_dialog.format_secondary_text(f"Invalid dataset folder")
                error_dialog.set_icon_name("dialog-error")
                error_dialog.run()
                error_dialog.destroy()
        dialog.destroy()
    
    def on_save_activate(self, menu_item):
        dialog = Gtk.FileChooserDialog(
            title="Select a folder to save the dataset",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
            buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "Save", Gtk.ResponseType.OK)
        )

        dialog.set_default_size(800, 400)
        dialog.set_current_folder(os.path.expanduser("~"))

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            folder_path = dialog.get_filename()
            print("Selected folder:", folder_path)
            self.save_ds(folder_path)
        dialog.destroy()
        
    def on_toggle_toggled(self, renderer, path):
        iter = self.liststore.get_iter(path)
        self.liststore[iter][1] = not self.liststore[iter][1]

    def on_show_fronts_toggled(self, menu_item):
        state = menu_item.get_active()
        self.display_fronts = state
        print("Fronts config changed")
        self.selected()

    def on_split_toggled(self, menu_item):
        state = menu_item.get_active()
        self.display_split = state
        print("Display config changed")
        self.selected()
        
    def on_show_model_toggled(self, menu_item):
        state = menu_item.get_active()
        self.display_data = state
        print("Data config changed")
        self.selected()

    def selected(self, treeview=None):
        if not treeview:
            treeview = self.treeview
        selection = self.treeview.get_selection()
        model, iter = selection.get_selected()
        if iter is not None:
            data_name = model[iter][0]
            data_include = model[iter][1]
            self.change_image(data_name, data_include)

    def white_to_transparency(self, img):
        x = np.asarray(img.convert('RGBA')).copy()
        x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
        return Image.fromarray(x)

    def change_image(self, data_name, data_include):
        data_image = None
        fronts_image = None

        if self.display_data:
            data_image = Image.open(self.opened_path + "/data/" + data_name)

        if self.display_fronts:
            fronts_image = Image.open(self.opened_path + "/fronts/" + data_name)

        if data_image or fronts_image:
            max_width = max(data_image.width if data_image else 0, fronts_image.width if fronts_image else 0)
            max_height = max(data_image.height if data_image else 0, fronts_image.height if fronts_image else 0)

            if self.display_split:
                factor = 2
                offset = max_width
            else:
                factor = 1
                offset = 0
                
            img = Image.new('RGBA', (max_width * factor, max_height), color='white')

            if data_image:
                img.paste(data_image, (0, 0))

            if fronts_image:
                transparent_fronts = self.white_to_transparency(fronts_image)
                img.paste(transparent_fronts, (offset, 0), transparent_fronts)
                
            self.load_image(img)
        self.image_label.set_text(data_name)

    def list_included_items(self):
        included_items = []
        excluded_items = []
        for row in self.liststore:
            if row[1]:
                included_items.append(row[0])
            else:
                excluded_items.append(row[0])
        return included_items, excluded_items

    def save_ds(self, folder):
        print("Saving dataset") 
        dialog = TestDatasetDialog(self)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            percentage = dialog.get_percentage()
        else: 
            percentage = 0
        dialog.destroy()
        print("Test dataset percentage:", percentage)
        train_dataset, excluded = self.list_included_items()
        test_dataset = random.sample(train_dataset, k=int(len(train_dataset) * int(percentage) / 100))
        total = len(train_dataset)
        progress = 0
        progress_dialog = ProgressDialog(self)
        for dst in ["/fronts-test", "/fronts", "/data", "/data-test"]:
            if not os.path.exists(folder + dst):
                os.makedirs(folder + dst)
        f = open(folder + "/exclude.list", "w")
        for item in excluded:
            f.write(os.path.splitext(item)[0] + "\n")
        f.close()
        for file in test_dataset:
            train_dataset.remove(file)
            fraction = progress / total
            progress_dialog.update_progress(fraction)
            shutil.copyfile(self.opened_path + "/fronts/" + file, folder + "/fronts-test/" + file)
            shutil.copyfile(self.opened_path + "/data/" + file, folder + "/data-test/" + file)
            progress += 1
        for file in train_dataset:
            fraction = progress / total
            progress_dialog.update_progress(fraction)
            shutil.copyfile(self.opened_path + "/fronts/" + file, folder + "/fronts/" + file)
            shutil.copyfile(self.opened_path + "/data/" + file, folder + "/data/" + file)
            progress += 1
        progress_dialog.destroy()
       
    def on_about_activate(self, button):
        dialog = Gtk.AboutDialog()
        dialog.set_program_name("Frontanim Dataset Inspector")
        dialog.set_version("1.0")
        dialog.set_comments("An application for fast checking generated frontanim datasets")
        dialog.set_website("https://pernicka.cz/frontanim")
        dialog.set_authors(["Pavel Pernička"])
        dialog.set_logo_icon_name("applications-system")
        dialog.run()
        dialog.destroy()

win = SimpleApp()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()