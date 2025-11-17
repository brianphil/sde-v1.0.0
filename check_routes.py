"""Check routes in database."""
import sqlite3

conn = sqlite3.connect('senga.db')
cursor = conn.cursor()

cursor.execute('SELECT route_id, vehicle_id, status, started_at, completed_at FROM routes')
print('[OK] Routes in database:')
for row in cursor.fetchall():
    print(f'  {row[0]}: vehicle={row[1]}, status={row[2]}, started={row[3]}, completed={row[4]}')

cursor.execute('SELECT stop_id, route_id, stop_type, status FROM route_stops')
print('\n[OK] Route Stops in database:')
for row in cursor.fetchall():
    print(f'  {row[0]}: route={row[1]}, type={row[2]}, status={row[3]}')

conn.close()
