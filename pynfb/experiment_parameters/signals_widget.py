import sys
from PyQt4 import QtGui, QtCore
from collections import OrderedDict

signals = [{'sSignalName': 'Signal1',
            'fBandpassLowHz': 1,
            'fBandpassHighHz': 10},
           {'sSignalName': 'Signal2',
            'fBandpassLowHz': 1,
            'fBandpassHighHz': 30}
           ]

default_signal = {'sSignalName': 'Unnamed Signal',
                  'fBandpassLowHz': 0,
                  'fBandpassHighHz': 250}

protocol_default = {'sProtocolName': 'Unnamed Protocol',
                    'bUpdateStatistics': False,
                    'fDuration': 10,
                    'fbSource': 'All',
                    'sFb_type': 'Baseline'}

protocols_types = ['Baseline', 'Circle']

protocols = [{'sProtocolName': 'Circle feedback',
              'fDuration': 102,
              'bUpdateStatistics': True,
              'fbSource': 'Signal2',
              'sFb_type': 'Circle'},
             {'sProtocolName': 'Baseline',
              'fDuration': 10,
              'bUpdateStatistics': True,
              'fbSource': 'All',
              'sFb_type': 'Baseline'}
             ]


protocols_sequence = ['Circle feedback']

parameters = {'vSignals': signals,
              'vProtocols': protocols,
              'vPSequence': protocols_sequence}

parameters_defaults = {'vSignals': default_signal,
                       'vProtocols': protocol_default}

class SignalsList(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = kwargs['parent'].params['vSignals']

        # layout
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        
        # label
        label = QtGui.QLabel('Signals:')
        layout.addWidget(label)

        # list of signals
        self.list = QtGui.QListWidget(self)
        self.reset_items()
        self.list.itemDoubleClicked.connect(self.item_double_clicked_event)
        layout.addWidget(self.list)

        # buttons layout
        buttons_layout = QtGui.QHBoxLayout()
        add_button = QtGui.QPushButton('Add')
        add_button.clicked.connect(self.add)
        remove_signal_button = QtGui.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.remove_current_item)
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)

    def add(self):
        self.params.append(default_signal.copy())
        self.reset_items()
        self.signals_dialogs[-1].open()

    def remove_current_item(self, item):
        current = self.list.currentRow()
        if current >= 0:
            del self.params[current]
            self.reset_items()

    def item_double_clicked_event(self, item):
        self.signals_dialogs[self.list.currentRow()].open()

    def reset_items(self):
        self.list.clear()
        self.signals_dialogs = []
        for signal in self.params:
            item = QtGui.QListWidgetItem(signal['sSignalName'])
            self.signals_dialogs.append(SignalDialog(self, signal_name=signal['sSignalName']))
            self.list.addItem(item)
        if self.list.currentRow() < 0:
            self.list.setItemSelected(self.list.item(0), True)


class SignalDialog(QtGui.QDialog):
    def __init__(self, parent, signal_name='Signal'):
        self.params = parent.params
        super().__init__(parent)
        self.parent_list = parent
        self.setWindowTitle('Properties: ' + signal_name)
        self.form_layout = QtGui.QFormLayout(self)
        # name
        self.name = QtGui.QLineEdit(self)
        self.name.setText(signal_name)
        self.form_layout.addRow('&Name:', self.name)
        # bandpass
        self.bandpass_low = QtGui.QSpinBox()
        self.bandpass_low.setRange(0, 250)
        self.bandpass_low.setValue(0)
        self.form_layout.addRow('&Bandpass low [Hz]:', self.bandpass_low)
        self.bandpass_high = QtGui.QSpinBox()
        self.bandpass_high.setRange(0, 250)
        self.bandpass_high.setValue(250)
        self.form_layout.addRow('&Bandpass high [Hz]:', self.bandpass_high)
        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def open(self):
        self.reset_items()
        super().open()

    def reset_items(self):
        self.bandpass_low.setValue(self.params[self.parent().list.currentRow()]['fBandpassLowHz'])
        self.bandpass_high.setValue(self.params[self.parent().list.currentRow()]['fBandpassHighHz'])

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sSignalName'] = self.name.text()
        self.params[current_signal_index]['fBandpassLowHz'] = self.bandpass_low.value()
        self.params[current_signal_index]['fBandpassHighHz'] = self.bandpass_high.value()
        self.parent().reset_items()
        self.close()


class ProtocolsList(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = kwargs['parent'].params['vProtocols']
        protocols_label = QtGui.QLabel('Protocols:')
        self.list = QtGui.QListWidget(self)

        self.list.setDragDropMode(QtGui.QAbstractItemView.DragOnly)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(protocols_label)
        layout.addWidget(self.list)
        self.reset_items()
        self.list.itemDoubleClicked.connect(self.item_double_clicked_event)
        buttons_layout = QtGui.QHBoxLayout()
        add_signal_button = QtGui.QPushButton('Add')
        add_signal_button.clicked.connect(self.add_action)
        remove_signal_button = QtGui.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.remove_current_item)
        buttons_layout.addWidget(add_signal_button)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        # self.show()

    def add_action(self):
        self.params.append(protocol_default.copy())
        self.reset_items()
        self.dialogs[-1].open()

    def remove_current_item(self, item):
        current = self.list.currentRow()
        if current >= 0:
            del self.params[current]
            self.reset_items()
        # self.show()

    def item_double_clicked_event(self, item):
        self.dialogs[self.list.currentRow()].open()

    def reset_items(self):
        self.list.clear()
        self.dialogs = []
        for signal in self.params:
            item = QtGui.QListWidgetItem(signal['sProtocolName'])
            self.dialogs.append(ProtocolDialog(self, protocol_name=signal['sProtocolName']))
            self.list.addItem(item)
        if self.list.currentRow() < 0:
            self.list.setItemSelected(self.list.item(0), True)


class ProtocolDialog(QtGui.QDialog):
    def __init__(self, parent, protocol_name='Protocol'):
        super().__init__(parent)
        self.params = parent.params
        self.parent_list = parent
        self.setWindowTitle('Properties: ' + protocol_name)
        self.form_layout = QtGui.QFormLayout(self)
        # name
        self.name = QtGui.QLineEdit(self)
        self.name.setText(protocol_name)
        self.form_layout.addRow('&Name:', self.name)
        # duration
        self.duration =  QtGui.QSpinBox()
        self.duration.setRange(0, 1000000)
        #self.duration.setValue(protocol_default['fDuration'])
        self.form_layout.addRow('&Duration [s]:', self.duration)
        # update statistics in the end
        self.update_statistics = QtGui.QCheckBox()
        #self.update_statistics.setTristate(protocol_default['bUpdateStatistics'])
        self.form_layout.addRow('&Update statistics:', self.update_statistics)
        # source signal
        self.source_signal = QtGui.QComboBox()
        self.update_combo_box()
        self.form_layout.addRow('&Source signal:', self.source_signal)
        # feedback type
        self.type = QtGui.QComboBox()
        for protocol_type in protocols_types:
            self.type.addItem(protocol_type)
        #self.type.setCurrentIndex(protocols_types.index(self.params))
        self.form_layout.addRow('&Type:', self.type)
        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def update_combo_box(self):
        self.source_signal.clear()
        for signal in signals:
            self.source_signal.addItem(signal['sSignalName'])
        self.source_signal.addItem('All')

    def open(self):
        self.update_combo_box()
        self.reset_items()
        super().open()

    def reset_items(self):
        current_protocol = self.params[self.parent().list.currentRow()]
        self.duration.setValue(current_protocol['fDuration'])
        self.update_statistics.setChecked(current_protocol['bUpdateStatistics'])
        self.source_signal.setCurrentIndex(
            self.source_signal.findText(current_protocol['fbSource'], QtCore.Qt.MatchFixedString))
        self.type.setCurrentIndex(
            self.type.findText(current_protocol['sFb_type'], QtCore.Qt.MatchFixedString))
        pass

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sProtocolName'] = self.name.text()
        self.params[current_signal_index]['fDuration'] = self.duration.value()
        self.params[current_signal_index]['bUpdateStatistics'] = self.update_statistics.isChecked()
        self.params[current_signal_index]['fbSource'] = self.source_signal.currentText()
        self.params[current_signal_index]['sFb_type'] = self.type.currentText()
        self.parent().reset_items()
        self.close()


class ProtocolSequenceList(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = kwargs['parent'].params['vPSequence']
        label = QtGui.QLabel('Protocols sequence:')
        self.list = ProtocolSequenceListWidget(parent=self)
        #self.list.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        #self.list.connect.
        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        buttons_layout = QtGui.QHBoxLayout()
        remove_signal_button = QtGui.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.list.remove_current_row)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)




class ProtocolSequenceListWidget(QtGui.QListWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = kwargs['parent'].params
        self.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.reset_items()

    def dropEvent(self, QDropEvent):
        super().dropEvent(QDropEvent)
        self.save()
        print(self.params)


    def reset_items(self):
        self.clear()
        for protocol in self.params:
            item = QtGui.QListWidgetItem(protocol)
            self.addItem(item)

    def save(self):
        self.params = [self.item(j).text() for j in range(self.count())]

    def remove_current_row(self):
        current = self.currentRow()
        if current >= 0:
            del self.params[current]
            self.reset_items()


class SettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = QtGui.QHBoxLayout()
        self.params = parameters.copy()
        self.protocols_list = ProtocolsList(parent=self)
        self.signals_list = SignalsList(parent=self)
        self.protocols_sequence_list = ProtocolSequenceList(parent=self)
        layout.addWidget(self.signals_list)
        layout.addWidget(self.protocols_list)
        layout.addWidget(self.protocols_sequence_list)
        self.setLayout(layout)
        self.show()

    def reset_parameters(self):
        self.signals_list.reset_items()
        self.protocols_list.reset_items()
        self.protocols_sequence_list.list.reset_items()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = SettingsWidget()
    window.show()
    sys.exit(app.exec_())