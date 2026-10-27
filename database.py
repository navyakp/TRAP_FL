import sqlite3, os
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'trapfl.db')
def get_db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = get_db()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS prediction_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_type TEXT NOT NULL,
            input_data TEXT,
            probability REAL,
            ids_classification TEXT,
            ips_action TEXT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    ''')
    c.commit(); c.close()
def get_user_by_id(uid):
    c=get_db(); r=c.execute('SELECT * FROM users WHERE id=?',(uid,)).fetchone(); c.close(); return dict(r) if r else None
def get_user_by_username(u):
    c=get_db(); r=c.execute('SELECT * FROM users WHERE username=?',(u,)).fetchone(); c.close(); return dict(r) if r else None
def create_user(username,email,ph):
    c=get_db()
    try: c.execute('INSERT INTO users(username,email,password_hash)VALUES(?,?,?)',(username,email,ph)); c.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: c.close()
def save_prediction(user_id,prediction_type,input_data,probability,ids_classification,ips_action,reason):
    c=get_db()
    c.execute('INSERT INTO prediction_history(user_id,prediction_type,input_data,probability,ids_classification,ips_action,reason)VALUES(?,?,?,?,?,?,?)',(user_id,prediction_type,input_data,probability,ids_classification,ips_action,reason))
    c.commit(); c.close()
def get_user_history(user_id,limit=200):
    c=get_db(); rows=c.execute('SELECT * FROM prediction_history WHERE user_id=? ORDER BY created_at DESC LIMIT ?',(user_id,limit)).fetchall(); c.close(); return [dict(r) for r in rows]
def delete_history_item(item_id,user_id):
    c=get_db(); c.execute('DELETE FROM prediction_history WHERE id=? AND user_id=?',(item_id,user_id)); c.commit(); c.close()
def clear_user_history(user_id):
    c=get_db(); c.execute('DELETE FROM prediction_history WHERE user_id=?',(user_id,)); c.commit(); c.close()
