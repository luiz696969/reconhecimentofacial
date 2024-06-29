from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, RoundedRectangle
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import os
import json

# Configurações iniciais da janela Kivy
Window.clearcolor = (1, 1, 1, 1)
Window.size = (980, 720)

# Caminho para salvar os dados dos usuários
data_path = 'L:/python-recognition-opencv-main/python-recognition-opencv-main/faces/'

# Função para salvar os dados do usuário
def salvar_dados_usuario(nome, cpf, senha):
    user_data = {
        'nome': nome,
        'cpf': cpf,
        'senha': senha
    }
    with open(os.path.join(data_path, f'{cpf}.json'), 'w') as f:
        json.dump(user_data, f)

# Widget Kivy para exibir a câmera
class KivyCV(Image):
    def __init__(self, capture, fps, **kwargs):
        super().__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(r'L:\python-recognition-opencv-main\.venv\Lib\haarcascade_frontalface_default.xml')
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            buf = cv2.flip(frame, 0).tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

# Aplicação principal Kivy
class Sistema(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(TelaInicial(name='telaInicial'))
        sm.add_widget(TelaFuncao(name='telaFuncao'))
        return sm

# Tela inicial
class TelaInicial(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        self.logo = Image(source="L:\python-recognition-opencv-main\python-recognition-opencv-main\images\icev_logo.png")
        self.logo.size_hint = (0.2, 0.2)
        self.logo.pos_hint = {'x': 0.4, 'top': 1}

        self.title = Label(text='Reconhecimento Facial', font_size='30sp', color=[0.031, 0.102, 0.227, 1],
                           font_name="L:/Oslo-Black.ttf", size_hint=(.99, .99), pos_hint={'x': .0, 'y': .29})

        FOTO = Button(text='FOTOS', on_press=self.ir_para_tela_funcao)
        CADASTRAR = Button(text='CADASTRAR', on_press=self.cadastrar_usuario)

        self.caixa_cpf = Label(text="CPF:", size_hint=(.005, .07), color=[0.467, 0.467, 0.467, 1],
                               font_size=26, pos_hint={'x': .21, 'y': .62})
        self.cpf = RoundedTextInput(multiline=False, size_hint=(.4, .07), font_size=26, pos_hint={'x': .30, 'y': .62})

        self.caixa_senha = Label(text="Senha:", size_hint=(.005, .07), color=[0.467, 0.467, 0.467, 1],
                                 font_size=26, pos_hint={'x': .21, 'y': .48})
        self.senha = RoundedTextInput(multiline=False, size_hint=(.4, .07), font_size=26, pos_hint={'x': .30, 'y': .48})

        self.caixa_nome = Label(text="Nome:", size_hint=(.005, .07), color=[0.467, 0.467, 0.467, 1],
                                font_size=26, pos_hint={'x': .21, 'y': .74})
        self.nome = RoundedTextInput(multiline=False, size_hint=(.4, .07), font_size=26, pos_hint={'x': .30, 'y': .74})

        box = BoxLayout(orientation='horizontal', size_hint=(0.4, 0.2), padding=8, pos_hint={'top': 0.2, 'center_x': 0.5})
        box.add_widget(CADASTRAR)
        box.add_widget(FOTO)

        layout.add_widget(box)
        layout.add_widget(self.logo)
        layout.add_widget(self.title)
        layout.add_widget(self.caixa_senha)
        layout.add_widget(self.senha)
        layout.add_widget(self.caixa_cpf)
        layout.add_widget(self.cpf)
        layout.add_widget(self.caixa_nome)
        layout.add_widget(self.nome)

        self.add_widget(layout)

    def ir_para_tela_funcao(self, instance):
        print('VOCÊ FOI PARA A TELA DE FUNÇÃO')
        self.manager.current = 'telaFuncao'

    def cadastrar_usuario(self, instance):
        senha = self.senha.text
        cpf = self.cpf.text
        nome = self.nome.text

        # Criar pasta com o nome do usuário
        user_dir = os.path.join(data_path, nome)
        os.makedirs(user_dir, exist_ok=True)

        # Salvar dados do usuário
        salvar_dados_usuario(nome, cpf, senha)

        print(f'Usuário cadastrado com CPF: {cpf} e senha: {senha}')

# Tela de funcionalidades
class TelaFuncao(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        self.botao_voltar = Button(text='VOLTAR')
        self.botao_voltar.size_hint = (.2, .1)
        self.botao_voltar.pos_hint = {'x': .55, 'y': .50}

        self.botao_tirar_fotos = Button(text='TIRAR FOTOS', on_press=self.capturar_faces)
        self.botao_tirar_fotos.size_hint = (.2, .1)
        self.botao_tirar_fotos.pos_hint = {'x': .25, 'y': .50}

        self.botao_voltar.bind(on_press=self.voltar)
        layout.add_widget(self.botao_voltar)
        layout.add_widget(self.botao_tirar_fotos)

        self.add_widget(layout)

    def voltar(self, *args):
        print('VOCÊ CLICOU NO BOTÃO VOLTAR')
        self.manager.current = 'telaInicial'

    def capturar_faces(self, *args):
        print('VOCÊ CLICOU NO BOTÃO TIRAR FOTOS')

        def extrair_face(img):
            face_cascade = cv2.CascadeClassifier(r'L:\python-recognition-opencv-main\.venv\Lib\haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print("Nenhuma face detectada!")
                return None

            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
                print(f"Face detectada nas coordenadas (x, y, w, h): {x}, {y}, {w}, {h}")

                # Salvar a face detectada na pasta do usuário
                cpf = self.manager.get_screen('telaInicial').cpf.text
                user_dir = os.path.join(data_path, cpf)
                os.makedirs(user_dir, exist_ok=True)
                file_name_path = os.path.join(user_dir, f'user_{len(os.listdir(user_dir))}.jpg')
                cv2.imwrite(file_name_path, cropped_face)

            return cropped_face

        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if ret:
                face = extrair_face(frame)
                if face is not None:
                    count += 1
                    face = cv2.resize(face, (200, 200))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                    cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', frame)

                else:
                    print("Face not Found")

            if cv2.waitKey(1) == 13 or count == 30:
                break

        cap.release()
        cv2.destroyAllWindows()
        print('Coleta de amostras completa!')

class RoundedTextInput(TextInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.858, 0.858, 0.858, 1)  # White background
            self.rect = RoundedRectangle(size=self.size, pos=self.pos, radius=[10])
            self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

if __name__ == '__main__':
    Sistema().run()
