// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLX

private let lock = NSLock()
nonisolated(unsafe) private var logs = [String: [String]]()

@Suite(.serialized) struct LoggingTests {

    class CollectHandler: MLXLogHandler, @unchecked (Sendable) {

        private let label: String
        public init(label: String) {
            self.label = label
        }

        func log(
            level: MLX.LogLevel, message: () -> String, metadata: () -> [String: String]?,
            file: StaticString, function: StaticString, line: UInt
        ) {
            let message = level.description + ": " + message()
            lock.withLock {
                logs[label, default: []].append(message)
            }
        }
    }

    @Test func testLoggingFactory() async throws {
        MLXLogger.factory = { CollectHandler(label: $0) }
        let logger = MLXLogger(label: "case1")

        logger.debug("test debug")
        logger.info("test info")
        logger.error("test error")

        let list = lock.withLock {
            logs["case1"] ?? []
        }

        #expect(
            list == [
                "debug: test debug",
                "info: test info",
                "error: test error",
            ])
    }

    #if canImport(OSLog)
        @Test func testOSLog() async throws {
            OSLogHandler.install()
            let logger = MLXLogger(label: "case2")

            // no assertions but should run without error
            logger.debug("test debug")
            logger.info("test info")
            logger.error("test error")

            // and not collect
            let list = lock.withLock {
                logs["case2"]
            }
            #expect(list == nil)
        }

        @Test func testOSLogKeepsWarningsApartFromErrors() {
            #expect(OSLogHandler.osLogType(for: .warning) == .default)
            #expect(OSLogHandler.osLogType(for: .error) == .error)
            #expect(OSLogHandler.osLogType(for: .info) == .info)
            #expect(OSLogHandler.osLogType(for: .debug) == .debug)
        }
    #endif

    @Test func testConcurrentFirstLogsShareOneHandler() async throws {
        // every thread logging through a new logger must get the same handler:
        // the factory runs once and every record reaches it
        let created = Counter()
        let previous = MLXLogger.factory
        defer { MLXLogger.factory = previous }
        MLXLogger.factory = { label in
            created.increment()
            return CollectHandler(label: label)
        }

        let logger = MLXLogger(label: "case4")
        let count = 64
        DispatchQueue.concurrentPerform(iterations: count) { i in
            logger.info("message \(i)")
        }

        #expect(created.value == 1)
        let list = lock.withLock { logs["case4"] ?? [] }
        #expect(list.count == count)
    }

    @Test func testStderrLineHasLabel() {
        let line = StderrHandler(label: "KVCache").formatted(
            level: .warning, message: "evicted", metadata: nil,
            date: Date(timeIntervalSince1970: 0))
        #expect(line.hasSuffix(" [warning] KVCache: evicted\n"))
    }

    @Test func testStderrLineHasMetadata() {
        let line = StderrHandler(label: "KVCache").formatted(
            level: .info, message: "evicted", metadata: ["tokens": "128", "layer": "3"],
            date: Date(timeIntervalSince1970: 0))
        #expect(line.hasSuffix(" [info] KVCache: evicted layer=3 tokens=128\n"))
        #expect(metadataSuffix([:]) == "")
    }

    @Test func testStderr() async throws {
        StderrHandler.install()
        let logger = MLXLogger(label: "case3")

        // no assertions but should run without error
        logger.debug("test debug")
        logger.info("test info")
        logger.error("test error")

        // and not collect
        let list = lock.withLock {
            logs["case3"]
        }
        #expect(list == nil)
    }

}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func increment() {
        lock.withLock { count += 1 }
    }

    var value: Int {
        lock.withLock { count }
    }
}
