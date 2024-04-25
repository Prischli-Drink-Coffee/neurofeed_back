import pymysql
import dotenv
import os

dotenv.load_dotenv()

class DataBase():
    def __init__(self, host: str = os.getenv("host"), user: str = os.getenv("user"), password: str = os.getenv("password"), database: str = os.getenv("database")):
        self.con = pymysql.connect(
            host=host, user=user, password=password, database=database)
        self.cur = self.con.cursor()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS vebinar
         (vebinar_id INT AUTO_INCREMENT PRIMARY KEY,
         name VARCHAR(255) UNIQUE NOT NULL) ''')
        self.cur.execute(
            '''CREATE TABLE IF NOT EXISTS users
             (user_id INT AUTO_INCREMENT PRIMARY KEY,
             username VARCHAR(255) UNIQUE NOT NULL,
             hashed_password VARCHAR(255))''')
        self.cur.execute(
            '''CREATE TABLE IF NOT EXISTS feed
             (feed_id INT AUTO_INCREMENT PRIMARY KEY,
             feedback VARCHAR(255),
             userful BOOL,
             emotion INT,
             keypoint VARCHAR(255),
             vebinar_id INT,
             user_id INT,
             FOREIGN KEY (vebinar_id) REFERENCES vebinar(vebinar_id),
             FOREIGN KEY (user_id) REFERENCES users(user_id)
             )''')
        self.cur.execute('''CREATE TABLE IF NOT EXISTS vebinar_user
         (vebinar_user_id INT AUTO_INCREMENT PRIMARY KEY, 
        vebinar_id INT,
         user_id INT,
         FOREIGN KEY (vebinar_id) REFERENCES vebinar(vebinar_id),
         FOREIGN KEY (user_id) REFERENCES users(user_id)) ''')

        self.con.commit()

    def add_user(self, username: str, hashed_password: str):
        self.cur.execute("INSERT INTO users (username, hashed_password) VALUES (%s,%s)",
                         (username, hashed_password))
        self.cur.execute("SELECT LAST_INSERT_ID()")
        self.con.commit()
        return (self.cur.fetchone())

    def add_vebinar(self, name: str):
        self.cur.execute("INSERT INTO vebinar (name) VALUES (%s)",
                         (name))
        self.cur.execute("SELECT LAST_INSERT_ID()")
        self.con.commit()
        return (self.cur.fetchone())

    def add_feed(self, feedback: str, userful: bool, emotion: int, keypoint: str, vebinar_id: int, user_id: int):
        self.cur.execute("INSERT INTO feed (feedback, userful, emotion, keypoint, vebinar_id, user_id) VALUES (%s, %s, %s, %s, %s, %s)",
                         (feedback, userful, emotion, keypoint, vebinar_id, user_id))
        self.cur.execute("SELECT LAST_INSERT_ID()")
        self.con.commit()
        return (self.cur.fetchone())

    def get_user_by_id(self, user_id: int):
        self.cur.execute(
            "SELECT * FROM users WHERE user_id = %s", (user_id))
        return (self.cur.fetchone())

    def get_feed_by_id(self, feed_id: int):
        self.cur.execute(
            "SELECT * FROM feed WHERE feed_id = %s", (feed_id))
        return (self.cur.fetchone())

    def get_vebinar_by_id(self, vebinar_id: int):
        self.cur.execute(
            "SELECT * FROM vebinar WHERE vebinar_id = %s", (vebinar_id))
        panel = [a for a in self.cur.fetchone()]
        return panel

    def get_feeds_from_one_user(self, user_id: int):
        self.cur.execute("SELECT * FROM feed WHERE user_id = %s", (user_id))
        return (self.cur.fetchone())

    def get_feeds_for_one_vebinar(self, vebinar_id: int):
        self.cur.execute(
            "SELECT * FROM feed WHERE vebinar_id = %s", (vebinar_id))
        return (self.cur.fetchone())

    def get_vebinar_all(self):
        self.cur.execute("SELECT * FROM vebinar")
        result = [[b for b in a] for a in self.cur.fetchall()]

        return (result)
