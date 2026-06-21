import { useEffect, useState } from 'react';

function App() {
  const [games, setGames] = useState([]);

  useEffect(() => {
    // In reality, this would connect to the websocket
    fetch('http://localhost:8000/api/games/active')
      .then(res => res.json())
      .then(data => setGames(data))
      .catch(err => console.error("Error fetching games", err));
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-blue-400">Quoridor Training Platform</h1>
        <p className="text-gray-400 mt-2">Live Monitor & Dashboard</p>
      </header>

      <main>
        <section className="bg-gray-800 rounded-lg p-6 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">Active Games</h2>
          {games.length === 0 ? (
            <p className="text-gray-400 italic">No games currently active. Start the worker loop.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {games.map((g: any) => (
                <div key={g.id} className="bg-gray-700 p-4 rounded-md border border-gray-600">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-sm bg-blue-600 px-2 py-1 rounded text-white">{g.mode}</span>
                    <span className="text-xs text-green-400 font-mono">ID: {g.id.substring(0, 8)}...</span>
                  </div>
                  <p className="text-gray-300"><span className="text-white font-semibold">{g.player_1}</span> vs <span className="text-white font-semibold">{g.player_2}</span></p>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
