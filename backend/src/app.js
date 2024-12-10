const express = require('express');
const routes = require('./routes/index');

const app = express();
const PORT = 3000;

// Middleware
app.use(express.json()); // For parsing JSON bodies

// Routes
app.use('/api', routes);

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
