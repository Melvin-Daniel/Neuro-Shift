// EMG stream for neuro-shift: two AD8232 outputs on A0 (jaw) and A1 (eyebrow/frontalis).
// delay(10) targets ~100 Hz (live + training must use same sketch).
// At 9600 baud, throughput is tight; keep delay(10) unless you retune window_size in Python.
//
// Wiring: both modules VCC->5V, GND->GND. OUT1->A0, OUT2->A1.
// LO+/LO- (leads-off detection inputs):
// - Module 1: LO+ -> D10, LO- -> D11
// - Module 2: LO+ -> D12, LO- -> D13
// Note: this sketch currently does NOT read LO pins; it only streams A0/A1.
void setup() {
  Serial.begin(9600);
  pinMode(10, INPUT); // LO+ (module 1)
  pinMode(11, INPUT); // LO- (module 1)
  pinMode(12, INPUT); // LO+ (module 2)
  pinMode(13, INPUT); // LO- (module 2)
}

void loop() {
  int jaw = analogRead(A0);
  int brow = analogRead(A1);
  Serial.print(millis());
  Serial.print(",");
  Serial.print(jaw);
  Serial.print(",");
  Serial.println(brow);
  delay(10);
}