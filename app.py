from flask import Flask, render_template, redirect, url_for, request, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import os
from sqlalchemy import func


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tomato_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# Models


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    settings = db.relationship('UserSettings', backref='user', uselist=False, cascade="all, delete-orphan")
    recognitions = db.relationship('TomatoRecognition', backref='user', lazy=True, cascade="all, delete-orphan")
    cameras = db.relationship('Camera', backref='user', lazy=True, cascade="all, delete-orphan")

class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    desired_ripeness = db.Column(db.String(20), default='both')  # "red", "yellow" или "both"
    conf_threshold = db.Column(db.Float, default=0.8)
    max_harvest_time = db.Column(db.Float, default=7.0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)

class TomatoRecognition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.current_timestamp())
    tomato_id = db.Column(db.Integer)  # ID внутри потока распознавания
    mean_h = db.Column(db.Float)
    mean_s = db.Column(db.Float)
    mean_v = db.Column(db.Float)
    mean_L = db.Column(db.Float)
    mean_a = db.Column(db.Float)
    mean_b = db.Column(db.Float)
    ripeness_percentage = db.Column(db.Float)
    classification = db.Column(db.String(20))
    is_ripe = db.Column(db.String(5))  # "Yes" или "No"
    time_to_harvest = db.Column(db.Float)
    collected = db.Column(db.String(20))  # оставляем пустым
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'))

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    source = db.Column(db.String(100))  # индекс камеры или URL
    active = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    recognitions = db.relationship('TomatoRecognition', backref='camera', lazy=True, cascade="all, delete-orphan")


# Flask-Login


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Routes: Registration & Login


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash("Username already exists!")
            return redirect(url_for('register'))
        new_user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        settings = UserSettings(user_id=new_user.id)
        db.session.add(settings)
        db.session.commit()
        flash("Registration successful! Please login.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials!")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# Settings and Camera Management


@app.route('/settings', methods=['GET','POST'])
@login_required
def settings():
    if request.method == 'POST':
        desired_ripeness = request.form['desired_ripeness']
        conf_threshold = float(request.form['conf_threshold'])
        max_harvest_time = float(request.form['max_harvest_time'])
        current_user.settings.desired_ripeness = desired_ripeness
        current_user.settings.conf_threshold = conf_threshold
        current_user.settings.max_harvest_time = max_harvest_time
        db.session.commit()
        flash("Settings updated!")
        return redirect(url_for('dashboard'))
    return render_template('settings.html', settings=current_user.settings)

@app.route('/cameras', methods=['GET','POST'])
@login_required
def cameras():
    if request.method == 'POST':
        name = request.form['name']
        source = request.form['source']
        cam = Camera(name=name, source=source, user_id=current_user.id)
        db.session.add(cam)
        db.session.commit()
        flash("Camera added!")
        return redirect(url_for('cameras'))
    cams = Camera.query.filter_by(user_id=current_user.id).all()
    return render_template('cameras.html', cams=cams)

@app.route('/camera/<int:cam_id>')
@login_required
def camera_page(cam_id):
    cam = Camera.query.filter_by(id=cam_id, user_id=current_user.id).first()
    if not cam:
        flash("Camera not found!")
        return redirect(url_for('cameras'))
    # Если камера не активна, показываем сообщение
    if not cam.active:
        return render_template('camera.html', cam=cam, error="Камера не найдена")
    return render_template('camera.html', cam=cam)


# Video Feed and Recognition




@app.route('/video_feed')
@login_required
def video_feed():
    from Detector import gen_frames  # Избегаем циклического импорта
    return Response(gen_frames(current_user.settings, current_user.id, camera_id=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Statistics Page


@app.route('/stats', methods=['GET'])
@login_required
def stats():
    # Получаем фильтр из параметров запроса (default - all)
    filter_class = request.args.get('filter_class', 'all')  # all, Ripe, Yellow, Unripe

    red_count = db.session.query(func.count(TomatoRecognition.id))\
        .filter_by(user_id=current_user.id, classification="Ripe").scalar() or 0
    yellow_count = db.session.query(func.count(TomatoRecognition.id))\
        .filter_by(user_id=current_user.id, classification="Yellow").scalar() or 0
    unripe_count = db.session.query(func.count(TomatoRecognition.id))\
        .filter_by(user_id=current_user.id, classification="Unripe").scalar() or 0
    total_ripe = red_count + yellow_count

    # Для таблицы получаем записи в зависимости от фильтра
    if filter_class == 'all':
        records = TomatoRecognition.query.filter_by(user_id=current_user.id).all()
    else:
        records = TomatoRecognition.query.filter_by(user_id=current_user.id, classification=filter_class).all()

    return render_template('stats.html',
                           red_count=red_count,
                           yellow_count=yellow_count,
                           unripe_count=unripe_count,
                           total_ripe=total_ripe,
                           records=records,
                           filter_class=filter_class)

# Dashboard and Index

@app.route('/dashboard')
@login_required
def dashboard():
    total = TomatoRecognition.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', total=total)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Run App


def create_database():
    """ Создаёт базу данных, если её нет """
    if not os.path.exists("tomato_app.db"):
        with app.app_context():
            db.create_all()
        print("Database created!")

if __name__ == '__main__':
    create_database()
    app.run(debug=True)
