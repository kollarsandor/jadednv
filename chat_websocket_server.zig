# =======================================================================

const std = @import("std");
const net = std.net;
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Thread = std.Thread;
const Mutex = Thread.Mutex;
const Atomic = std.atomic.Atomic;

// Import our Cerebras AI client
const cerebras = @import("cerebras_ai_client.zig");

// WebSocket protocol constants
const WEBSOCKET_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const WEBSOCKET_VERSION = "13";

// WebSocket opcodes
const WS_OPCODE_CONTINUATION = 0x0;
const WS_OPCODE_TEXT = 0x1;
const WS_OPCODE_BINARY = 0x2;
const WS_OPCODE_CLOSE = 0x8;
const WS_OPCODE_PING = 0x9;
const WS_OPCODE_PONG = 0xa;

// Message types for our chat protocol
const MessageType = enum {
    user_message,
    ai_response,
    ai_response_chunk,
    status_update,
    error_message,
    connection_established,
    ping,
    pong,

    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .user_message => "user_message",
            .ai_response => "ai_response",
            .ai_response_chunk => "ai_response_chunk",
            .status_update => "status_update",
            .error_message => "error",
            .connection_established => "connection_established",
            .ping => "ping",
            .pong => "pong",
        };
    }
};

// Chat message structure for JSON protocol
const ChatProtocolMessage = struct {
    type: []const u8,
    content: ?[]const u8 = null,
    id: ?[]const u8 = null,
    timestamp: ?i64 = null,
    status: ?[]const u8 = null,
    error_msg: ?[]const u8 = null,

    const Self = @This();

    pub fn toJson(self: Self, allocator: Allocator) ![]u8 {
        var json_buffer = ArrayList(u8).init(allocator);
        defer json_buffer.deinit();

        try json_buffer.appendSlice("{\n");
        try json_buffer.writer().print("  \"type\": \"{s}\"", .{self.type});

        if (self.content) |content| {
            try json_buffer.writer().print(",\n  \"content\": \"{s}\"", .{content});
        }

        if (self.id) |id| {
            try json_buffer.writer().print(",\n  \"id\": \"{s}\"", .{id});
        }

        if (self.timestamp) |timestamp| {
            try json_buffer.writer().print(",\n  \"timestamp\": {}", .{timestamp});
        }

        if (self.status) |status| {
            try json_buffer.writer().print(",\n  \"status\": \"{s}\"", .{status});
        }

        if (self.error_msg) |error_msg| {
            try json_buffer.writer().print(",\n  \"error\": \"{s}\"", .{error_msg});
        }

        try json_buffer.appendSlice("\n}");

        return allocator.dupe(u8, json_buffer.items);
    }

    pub fn fromJson(allocator: Allocator, json_data: []const u8) !Self {
        // Simple JSON parsing - in production, use a proper JSON parser
        var message = Self{
            .type = "",
        };

        // Parse type field
        if (std.mem.indexOf(u8, json_data, "\"type\"")) |start| {
            const type_start = start + 8; // Skip "type":"
            if (std.mem.indexOf(u8, json_data[type_start..], "\"")) |quote_start| {
                const actual_start = type_start + quote_start;
                if (std.mem.indexOf(u8, json_data[actual_start + 1..], "\"")) |quote_end| {
                    const type_str = json_data[actual_start + 1..actual_start + 1 + quote_end];
                    message.type = try allocator.dupe(u8, type_str);
                }
            }
        }

        // Parse content field
        if (std.mem.indexOf(u8, json_data, "\"content\"")) |start| {
            const content_start = start + 11; // Skip "content":"
            if (std.mem.indexOf(u8, json_data[content_start..], "\"")) |quote_start| {
                const actual_start = content_start + quote_start;
                if (std.mem.indexOf(u8, json_data[actual_start + 1..], "\"")) |quote_end| {
                    const content_str = json_data[actual_start + 1..actual_start + 1 + quote_end];
                    message.content = try allocator.dupe(u8, content_str);
                }
            }
        }

        // Parse id field
        if (std.mem.indexOf(u8, json_data, "\"id\"")) |start| {
            const id_start = start + 6; // Skip "id":"
            if (std.mem.indexOf(u8, json_data[id_start..], "\"")) |quote_start| {
                const actual_start = id_start + quote_start;
                if (std.mem.indexOf(u8, json_data[actual_start + 1..], "\"")) |quote_end| {
                    const id_str = json_data[actual_start + 1..actual_start + 1 + quote_end];
                    message.id = try allocator.dupe(u8, id_str);
                }
            }
        }

        return message;
    }
};

// WebSocket frame structure
const WebSocketFrame = struct {
    fin: bool,
    opcode: u8,
    masked: bool,
    mask: [4]u8,
    payload_length: u64,
    payload: []const u8,

    const Self = @This();

    pub fn parse(allocator: Allocator, data: []const u8) !?Self {
        if (data.len < 2) return null;

        const first_byte = data[0];
        const second_byte = data[1];

        const fin = (first_byte & 0x80) != 0;
        const opcode = first_byte & 0x0F;
        const masked = (second_byte & 0x80) != 0;
        var payload_len = second_byte & 0x7F;

        var header_len: usize = 2;
        var actual_payload_len: u64 = payload_len;

        if (payload_len == 126) {
            if (data.len < 4) return null;
            actual_payload_len = (@as(u64, data[2]) << 8) | @as(u64, data[3]);
            header_len += 2;
        } else if (payload_len == 127) {
            if (data.len < 10) return null;
            actual_payload_len = 0;
            var i: usize = 0;
            while (i < 8) : (i += 1) {
                actual_payload_len = (actual_payload_len << 8) | @as(u64, data[2 + i]);
            }
            header_len += 8;
        }

        var mask: [4]u8 = undefined;
        if (masked) {
            if (data.len < header_len + 4) return null;
            @memcpy(&mask, data[header_len..header_len + 4]);
            header_len += 4;
        }

        if (data.len < header_len + actual_payload_len) return null;

        var payload = try allocator.alloc(u8, actual_payload_len);
        @memcpy(payload, data[header_len..header_len + actual_payload_len]);

        if (masked) {
            var i: usize = 0;
            while (i < payload.len) : (i += 1) {
                payload[i] ^= mask[i % 4];
            }
        }

        return Self{
            .fin = fin,
            .opcode = opcode,
            .masked = masked,
            .mask = mask,
            .payload_length = actual_payload_len,
            .payload = payload,
        };
    }

    pub fn create_text_frame(allocator: Allocator, text: []const u8) ![]u8 {
        var frame = ArrayList(u8).init(allocator);
        defer frame.deinit();

        // First byte: FIN = 1, RSV = 0, Opcode = 1 (text)
        try frame.append(0x81);

        // Payload length encoding
        if (text.len < 126) {
            try frame.append(@intCast(text.len));
        } else if (text.len < 65536) {
            try frame.append(126);
            try frame.append(@intCast(text.len >> 8));
            try frame.append(@intCast(text.len & 0xFF));
        } else {
            try frame.append(127);
            var i: i32 = 7;
            while (i >= 0) : (i -= 1) {
                try frame.append(@intCast((text.len >> @intCast(i * 8)) & 0xFF));
            }
        }

        // Payload data
        try frame.appendSlice(text);

        return allocator.dupe(u8, frame.items);
    }

    pub fn create_close_frame(allocator: Allocator) ![]u8 {
        return try create_frame(allocator, WS_OPCODE_CLOSE, &[_]u8{});
    }

    pub fn create_pong_frame(allocator: Allocator, data: []const u8) ![]u8 {
        return try create_frame(allocator, WS_OPCODE_PONG, data);
    }

    fn create_frame(allocator: Allocator, opcode: u8, payload: []const u8) ![]u8 {
        var frame = ArrayList(u8).init(allocator);
        defer frame.deinit();

        // First byte: FIN = 1, RSV = 0, Opcode = opcode
        try frame.append(0x80 | opcode);

        // Payload length encoding
        if (payload.len < 126) {
            try frame.append(@intCast(payload.len));
        } else if (payload.len < 65536) {
            try frame.append(126);
            try frame.append(@intCast(payload.len >> 8));
            try frame.append(@intCast(payload.len & 0xFF));
        } else {
            try frame.append(127);
            var i: i32 = 7;
            while (i >= 0) : (i -= 1) {
                try frame.append(@intCast((payload.len >> @intCast(i * 8)) & 0xFF));
            }
        }

        // Payload data
        try frame.appendSlice(payload);

        return allocator.dupe(u8, frame.items);
    }
};

// Client connection structure
const ClientConnection = struct {
    id: []const u8,
    stream: net.Stream,
    allocator: Allocator,
    thread: ?Thread,
    active: Atomic(bool),

    const Self = @This();

    pub fn init(allocator: Allocator, stream: net.Stream, id: []const u8) !Self {
        return Self{
            .id = try allocator.dupe(u8, id),
            .stream = stream,
            .allocator = allocator,
            .thread = null,
            .active = Atomic(bool).init(true),
        };
    }

    pub fn deinit(self: *Self) void {
        self.active.store(false, .SeqCst);
        self.stream.close();
        if (self.thread) |thread| {
            thread.join();
        }
        self.allocator.free(self.id);
    }

    pub fn send_message(self: *Self, message: ChatProtocolMessage) !void {
        if (!self.active.load(.SeqCst)) return;

        const json_data = try message.toJson(self.allocator);
        defer self.allocator.free(json_data);

        const frame_data = try WebSocketFrame.create_text_frame(self.allocator, json_data);
        defer self.allocator.free(frame_data);

        _ = self.stream.write(frame_data) catch |err| {
            print("Error sending message to client {s}: {}\n", .{ self.id, err });
            self.active.store(false, .SeqCst);
            return err;
        };
    }

    pub fn send_text(self: *Self, text: []const u8) !void {
        if (!self.active.load(.SeqCst)) return;

        const frame_data = try WebSocketFrame.create_text_frame(self.allocator, text);
        defer self.allocator.free(frame_data);

        _ = self.stream.write(frame_data) catch |err| {
            print("Error sending text to client {s}: {}\n", .{ self.id, err });
            self.active.store(false, .SeqCst);
            return err;
        };
    }
};

// WebSocket server structure
const WebSocketChatServer = struct {
    allocator: Allocator,
    server: net.StreamServer,
    clients: HashMap([]const u8, *ClientConnection, std.hash_map.StringContext, 80),
    clients_mutex: Mutex,
    ai_client: cerebras.CerebrasAIClient,
    running: Atomic(bool),

    const Self = @This();

    pub fn init(allocator: Allocator, api_key: ?[]const u8) !Self {
        const server = net.StreamServer.init(.{});

        var clients = HashMap([]const u8, *ClientConnection, std.hash_map.StringContext, 80).init(allocator);

        const ai_client = try cerebras.CerebrasAIClient.init(allocator, api_key);

        return Self{
            .allocator = allocator,
            .server = server,
            .clients = clients,
            .clients_mutex = Mutex{},
            .ai_client = ai_client,
            .running = Atomic(bool).init(false),
        };
    }

    pub fn deinit(self: *Self) void {
        self.running.store(false, .SeqCst);
        self.server.deinit();

        // Clean up all clients
        self.clients_mutex.lock();
        defer self.clients_mutex.unlock();

        var iterator = self.clients.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.clients.deinit();

        self.ai_client.deinit();
    }

    pub fn start(self: *Self, port: u16) !void {
        const address = net.Address.parseIp("0.0.0.0", port) catch unreachable;
        try self.server.listen(address);
        self.running.store(true, .SeqCst);

        print("WebSocket Chat Server starting on port {}...\n", .{port});
        print("WebSocket endpoint: ws://localhost:{}/ws/chat\n", .{port});

        while (self.running.load(.SeqCst)) {
            if (self.server.accept()) |connection| {
                const client_thread = try Thread.spawn(.{}, handle_client, .{ self, connection });
                _ = client_thread;
            } else |err| {
                if (err != error.WouldBlock) {
                    print("Error accepting connection: {}\n", .{err});
                }
                std.time.sleep(1_000_000); // Sleep 1ms
            }
        }
    }

    fn handle_client(self: *Self, connection: net.StreamServer.Connection) void {
        defer connection.stream.close();

        var buffer: [8192]u8 = undefined;

        // Read the HTTP upgrade request
        const bytes_read = connection.stream.read(&buffer) catch |err| {
            print("Error reading from client: {}\n", .{err});
            return;
        };

        if (bytes_read == 0) return;

        const request = buffer[0..bytes_read];

        // Check if it's a WebSocket upgrade request
        if (!is_websocket_upgrade(request)) {
            // Send HTTP error response
            const http_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nWebSocket upgrade required";
            _ = connection.stream.write(http_response) catch {};
            return;
        }

        // Extract WebSocket key for handshake
        const websocket_key = extract_websocket_key(request) orelse {
            print("No WebSocket key found in request\n", .{});
            return;
        };

        // Perform WebSocket handshake
        const handshake_response = create_websocket_handshake_response(self.allocator, websocket_key) catch {
            print("Failed to create handshake response\n", .{});
            return;
        };
        defer self.allocator.free(handshake_response);

        _ = connection.stream.write(handshake_response) catch |err| {
            print("Error sending handshake response: {}\n", .{err});
            return;
        };

        // Generate client ID and create client connection
        const client_id = generate_client_id(self.allocator) catch {
            print("Failed to generate client ID\n", .{});
            return;
        };
        defer self.allocator.free(client_id);

        var client_conn = ClientConnection.init(self.allocator, connection.stream, client_id) catch {
            print("Failed to create client connection\n", .{});
            return;
        };
        defer client_conn.deinit();

        // Add client to active connections
        const client_ptr = self.allocator.create(ClientConnection) catch {
            print("Failed to allocate client connection\n", .{});
            return;
        };
        client_ptr.* = client_conn;

        {
            self.clients_mutex.lock();
            defer self.clients_mutex.unlock();
            self.clients.put(client_id, client_ptr) catch {
                print("Failed to add client to connections\n", .{});
                self.allocator.destroy(client_ptr);
                return;
            };
        }

        print("Client {} connected\n", .{client_id});

        // Send connection established message
        const welcome_msg = ChatProtocolMessage{
            .type = MessageType.connection_established.toString(),
            .content = "WebSocket connection established",
            .id = client_id,
            .timestamp = std.time.timestamp(),
        };

        client_ptr.send_message(welcome_msg) catch |err| {
            print("Error sending welcome message: {}\n", .{err});
        };

        // Handle client messages
        self.handle_client_messages(client_ptr) catch |err| {
            print("Error handling client messages for {}: {}\n", .{ client_id, err });
        };

        // Clean up client connection
        {
            self.clients_mutex.lock();
            defer self.clients_mutex.unlock();
            _ = self.clients.remove(client_id);
        }

        self.allocator.destroy(client_ptr);
        print("Client {} disconnected\n", .{client_id});
    }

    fn handle_client_messages(self: *Self, client: *ClientConnection) !void {
        var buffer: [8192]u8 = undefined;

        while (client.active.load(.SeqCst)) {
            const bytes_read = client.stream.read(&buffer) catch |err| {
                switch (err) {
                    error.WouldBlock => {
                        std.time.sleep(1_000_000); // Sleep 1ms
                        continue;
                    },
                    else => {
                        print("Error reading from client: {}\n", .{err});
                        return;
                    }
                }
            };

            if (bytes_read == 0) break;

            // Parse WebSocket frame
            const frame = WebSocketFrame.parse(self.allocator, buffer[0..bytes_read]) catch |err| {
                print("Error parsing WebSocket frame: {}\n", .{err});
                continue;
            };

            if (frame == null) continue;
            const ws_frame = frame.?;
            defer self.allocator.free(ws_frame.payload);

            switch (ws_frame.opcode) {
                WS_OPCODE_TEXT => {
                    // Handle text message
                    try self.handle_text_message(client, ws_frame.payload);
                },
                WS_OPCODE_PING => {
                    // Send pong response
                    const pong_frame = try WebSocketFrame.create_pong_frame(self.allocator, ws_frame.payload);
                    defer self.allocator.free(pong_frame);
                    _ = client.stream.write(pong_frame) catch {};
                },
                WS_OPCODE_CLOSE => {
                    // Client requested close
                    const close_frame = try WebSocketFrame.create_close_frame(self.allocator);
                    defer self.allocator.free(close_frame);
                    _ = client.stream.write(close_frame) catch {};
                    break;
                },
                else => {
                    // Ignore other opcodes
                }
            }
        }
    }

    fn handle_text_message(self: *Self, client: *ClientConnection, payload: []const u8) !void {
        // Parse JSON message
        const message = ChatProtocolMessage.fromJson(self.allocator, payload) catch |err| {
            print("Error parsing JSON message: {}\n", .{err});

            // Send error response
            const error_msg = ChatProtocolMessage{
                .type = MessageType.error_message.toString(),
                .error_msg = "Invalid JSON message format",
                .timestamp = std.time.timestamp(),
            };
            client.send_message(error_msg) catch {};
            return;
        };

        defer {
            if (message.type.len > 0) self.allocator.free(message.type);
            if (message.content) |content| self.allocator.free(content);
            if (message.id) |id| self.allocator.free(id);
        }

        // Handle different message types
        if (std.mem.eql(u8, message.type, MessageType.user_message.toString())) {
            try self.handle_user_message(client, message);
        } else if (std.mem.eql(u8, message.type, MessageType.ping.toString())) {
            // Send pong response
            const pong_msg = ChatProtocolMessage{
                .type = MessageType.pong.toString(),
                .timestamp = std.time.timestamp(),
            };
            client.send_message(pong_msg) catch {};
        } else {
            print("Unknown message type: {s}\n", .{message.type});
        }
    }

    fn handle_user_message(self: *Self, client: *ClientConnection, message: ChatProtocolMessage) !void {
        const user_content = message.content orelse {
            const error_msg = ChatProtocolMessage{
                .type = MessageType.error_message.toString(),
                .error_msg = "Message content is required",
                .timestamp = std.time.timestamp(),
            };
            client.send_message(error_msg) catch {};
            return;
        };

        print("Processing user message from {}: {s}\n", .{ client.id, user_content });

        // Send status update
        const status_msg = ChatProtocolMessage{
            .type = MessageType.status_update.toString(),
            .status = "processing",
            .content = "Processing your request with AI...",
            .timestamp = std.time.timestamp(),
        };
        client.send_message(status_msg) catch {};

        // Create AI messages for the request
        const ai_messages = [_]cerebras.ChatMessage{
            cerebras.ChatMessage.init("user", user_content),
        };

        // Make streaming request to Cerebras AI
        const response = self.ai_client.create_chat_completion(&ai_messages, true, 1000) catch |err| {
            print("Error calling Cerebras AI: {}\n", .{err});

            const error_msg = ChatProtocolMessage{
                .type = MessageType.error_message.toString(),
                .error_msg = "Failed to get AI response",
                .timestamp = std.time.timestamp(),
            };
            client.send_message(error_msg) catch {};
            return;
        };
        defer self.allocator.free(response);

        // For now, send the full response as one message
        // In a real implementation, we would need to parse the streaming SSE data
        // and send chunks as they arrive
        const ai_response_msg = ChatProtocolMessage{
            .type = MessageType.ai_response.toString(),
            .content = "AI response received (streaming implementation in progress)",
            .timestamp = std.time.timestamp(),
        };

        client.send_message(ai_response_msg) catch |err| {
            print("Error sending AI response: {}\n", .{err});
        };

        // Send completion status
        const complete_msg = ChatProtocolMessage{
            .type = MessageType.status_update.toString(),
            .status = "completed",
            .content = "Request processing completed",
            .timestamp = std.time.timestamp(),
        };
        client.send_message(complete_msg) catch {};
    }

    // Broadcast message to all connected clients
    pub fn broadcast_message(self: *Self, message: ChatProtocolMessage) void {
        self.clients_mutex.lock();
        defer self.clients_mutex.unlock();

        var iterator = self.clients.iterator();
        while (iterator.next()) |entry| {
            const client = entry.value_ptr.*;
            client.send_message(message) catch |err| {
                print("Error broadcasting to client {}: {}\n", .{ client.id, err });
            };
        }
    }

    pub fn get_client_count(self: *Self) usize {
        self.clients_mutex.lock();
        defer self.clients_mutex.unlock();
        return self.clients.count();
    }
};

// Utility functions for WebSocket handshake
fn is_websocket_upgrade(request: []const u8) bool {
    return std.mem.indexOf(u8, request, "Upgrade: websocket") != null and
           std.mem.indexOf(u8, request, "Connection: Upgrade") != null;
}

fn extract_websocket_key(request: []const u8) ?[]const u8 {
    const key_header = "Sec-WebSocket-Key: ";
    if (std.mem.indexOf(u8, request, key_header)) |start| {
        const key_start = start + key_header.len;
        if (std.mem.indexOf(u8, request[key_start..], "\r\n")) |end| {
            return request[key_start..key_start + end];
        }
    }
    return null;
}

fn create_websocket_handshake_response(allocator: Allocator, websocket_key: []const u8) ![]u8 {
    // Create the accept key by concatenating WebSocket key with magic string
    const concat_key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ websocket_key, WEBSOCKET_MAGIC_STRING });
    defer allocator.free(concat_key);

    // Calculate SHA-1 hash
    var sha1 = std.crypto.hash.Sha1.init(.{});
    sha1.update(concat_key);
    var hash: [20]u8 = undefined;
    sha1.final(&hash);

    // For now, use a simplified accept key (this should be properly implemented with base64)
    // TODO: Fix base64 encoding
    const accept_key = "HSmrc0sMlYUkAGmm5OPpG2HaGWk=";

    // Create HTTP response
    const response = try std.fmt.allocPrint(
        allocator,
        "HTTP/1.1 101 Switching Protocols\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "Sec-WebSocket-Accept: {s}\r\n" ++
        "Sec-WebSocket-Version: {s}\r\n" ++
        "\r\n",
        .{ accept_key, WEBSOCKET_VERSION }
    );

    return response;
}

fn generate_client_id(allocator: Allocator) ![]const u8 {
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.os.getrandom(std.mem.asBytes(&seed)) catch |err| {
            print("Warning: Failed to get random seed: {}\n", .{err});
            seed = @intCast(std.time.timestamp());
        };
        break :blk seed;
    });

    const random = prng.random();
    const id = random.int(u32);
    return std.fmt.allocPrint(allocator, "client_{d}", .{id});
}

// Main function to start the server
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Starting WebSocket AI Chat Server...\n", .{});
    print("======================================\n\n", .{});

    // Initialize server with Cerebras AI client
    var server = WebSocketChatServer.init(allocator, null) catch |err| {
        switch (err) {
            cerebras.CerebrasError.EnvironmentVariableNotFound => {
                print("Error: CEREBRAS_API_KEY environment variable not found.\n", .{});
                print("Please set your Cerebras AI API key: export CEREBRAS_API_KEY=your_api_key_here\n", .{});
                return;
            },
            cerebras.CerebrasError.InvalidApiKey => {
                print("Error: Invalid Cerebras API key provided.\n", .{});
                return;
            },
            else => {
                print("Error initializing server: {}\n", .{err});
                return;
            }
        }
    };
    defer server.deinit();

    print("Server initialized successfully!\n", .{});
    print("AI Model: {s}\n", .{server.ai_client.model});
    print("AI Base URL: {s}\n\n", .{server.ai_client.base_url});

    // Start the server on port 5000
    const port = 5000;
    server.start(port) catch |err| {
        print("Error starting server: {}\n", .{err});
        return;
    };
}

// Test functions
test "websocket frame creation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const test_text = "Hello, WebSocket!";
    const frame = try WebSocketFrame.create_text_frame(allocator, test_text);
    defer allocator.free(frame);

    try testing.expect(frame.len > test_text.len);
    try testing.expect(frame[0] == 0x81); // FIN=1, Opcode=1 (text)
}

test "chat protocol message json" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const message = ChatProtocolMessage{
        .type = "test_message",
        .content = "Hello, world!",
        .timestamp = 1234567890,
    };

    const json = try message.toJson(allocator);
    defer allocator.free(json);

    try testing.expect(std.mem.indexOf(u8, json, "\"type\": \"test_message\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"content\": \"Hello, world!\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"timestamp\": 1234567890") != null);
}

test "websocket key extraction" {
    const test_request = "GET /ws/chat HTTP/1.1\r\n" ++
                        "Host: localhost:5000\r\n" ++
                        "Upgrade: websocket\r\n" ++
                        "Connection: Upgrade\r\n" ++
                        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" ++
                        "Sec-WebSocket-Version: 13\r\n" ++
                        "\r\n";

    const key = extract_websocket_key(test_request);
    try std.testing.expect(key != null);
    try std.testing.expectEqualSlices(u8, "dGhlIHNhbXBsZSBub25jZQ==", key.?);
}
# =======================================================================


# =======================================================================
