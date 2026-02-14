// Simple test script to verify file writing works
import fs from 'fs';
import path from 'path';

const outPath = path.resolve(process.cwd(), 'output');
const filePath = path.join(outPath, 'test.md');

console.log(`Attempting to write to: ${filePath}`);
console.log(`Output directory exists: ${fs.existsSync(outPath)}`);

try {
  fs.writeFileSync(filePath, '# Test File\n\nThis is a test.');
  console.log(`Successfully wrote file to ${filePath}`);
} catch (error) {
  console.error(`Failed to write file: ${error}`);
}
