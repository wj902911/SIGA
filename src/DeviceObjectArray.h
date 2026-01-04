#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "Utility_d.h"
#include <type_traits>

class KnotVector_d;
class TensorBsplineBasis_d;
template <typename T> class DeviceVector;
class Patch_d;
class GaussPoints_d;

template <typename T>
__global__ void deviceAssign(T* data, int size, const T* value);

template <typename T>
__global__ void deviceRead(T* value, int size, const T* data);

template <typename T>
__global__ void deviceDeepCopyKernel(T* device_dst, T* host_src);
__global__ void deviceDeepCopyKernel(KnotVector_d* device_dst, KnotVector_d* host_src);

template <typename T>
__global__ void destructKernel(T* ptr, size_t count);

__global__ void destructKernel(KnotVector_d* ptr, size_t count);
__global__ void destructKernel(TensorBsplineBasis_d* ptr, size_t count);
__global__ void destructKernel(DeviceVector<double>* ptr, size_t count);
__global__ void destructKernel(Patch_d* ptr, size_t count);
__global__ void destructKernel(GaussPoints_d* ptr, size_t count);


template <typename T>
class DeviceObjectArray
{
private:
    int m_size;
    T* m_data;
public:
    __host__ __device__
    DeviceObjectArray() : m_size(0), m_data(nullptr) {}

    __host__ __device__
    DeviceObjectArray(int size) : m_size(size)
    { 
        #if defined(__CUDA_ARCH__)
            m_data = new T[size]; 
        #else
            size_t sizeInBytes = static_cast<size_t>(size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray constructor");
        #endif
    }

    __host__ __device__
    DeviceObjectArray(int size, const T* data) : m_size(size)
    {
        if (size > 0) 
        {
        #if defined(__CUDA_ARCH__)
            m_data = new T[size];
            for (int i = 0; i < size; ++i) 
                m_data[i] = data[i];
        #else
            size_t sizeInBytes = static_cast<size_t>(size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray constructor");
            err = cudaMemcpy(m_data, data, sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                printf("cudaMemcpy failed in DeviceObjectArray constructor: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            } 
        #endif
        } 
        else 
            m_data = nullptr;
    }

    __host__
    DeviceObjectArray(const std::vector<T>& stdVector)
        : m_size(static_cast<int>(stdVector.size()))
    {
        if (m_size > 0) 
        {
            size_t sizeInBytes = static_cast<size_t>(m_size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray constructor");
            err = cudaMemcpy(m_data, stdVector.data(), sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray constructor: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            } 
        } 
        else 
            m_data = nullptr;
    }

    // Copy constructor
    __host__ __device__
    DeviceObjectArray(const DeviceObjectArray& other)
        : m_size(other.m_size)
    {
        if(m_size > 0) 
        {
        #if defined(__CUDA_ARCH__)
            m_data = static_cast<T*>(::operator new[](m_size * sizeof(T)));
            if (m_data == nullptr)
            {
                m_size = 0;
                return;
            }
            for (int i = 0; i < m_size; ++i) 
                new (&m_data[i]) T(other.m_data[i]);
            //printf("DeviceObjectArray copy constructor used on device!\n");
        #else
#if 1
    #if 0
            T* hostTemp = new T[m_size];
            for (int i = 0; i < m_size; ++i)
            {
                cudaError_t err = cudaMemcpy(&hostTemp[i],
                                             other.m_data + i,
                                             sizeof(T),
                                             cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess && "cudaMemcpy (device->host) failed in DeviceObjectArray copy constructor");
            }
            T* hostDeep = new T[m_size];
            for (int i = 0; i < m_size; ++i) 
            {
                if constexpr (std::is_trivially_copyable<T>::value) 
                    hostDeep[i] = hostTemp[i];
                else
                    hostDeep[i] = hostTemp[i].clone(); 
            } 
            size_t sizeInBytes = static_cast<size_t>(m_size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy constructor");
            err = cudaMemcpy(m_data, hostDeep, sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy constructor: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }

            delete[] hostTemp; // Free the temporary host array
            delete[] hostDeep; // Free the deep copy host array
    #else
        #if 1
            size_t sizeInBytes = static_cast<size_t>(m_size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy constructor");
            //err = cudaMemcpy(m_data, other.m_data, sizeInBytes, cudaMemcpyHostToDevice);
            err = cudaMemcpy(m_data, other.m_data, sizeInBytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy constructor: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }
        #else
            size_t sizeInBytes = static_cast<size_t>(m_size * sizeof(T));
            T* h_Data = new T[m_size];
            cudaError_t err = cudaMemcpy(h_Data, other.m_data, sizeInBytes, cudaMemcpyDeviceToHost);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy constructor Device->Host");
            err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy constructor Host->Device");
            err = cudaMemcpy(m_data, h_Data, sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy constructor: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }
            delete [] h_Data;
        #endif
    #endif
            
#else
            T* hostDeep = new T[m_size];
            for (int i = 0; i < m_size; ++i)
            {
                T shallow;
                cudaError_t err = cudaMemcpy(&shallow,
                                             other.m_data + i,
                                             sizeof(T),
                                             cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess && 
                "cudaMemcpy (device->host) failed in DeviceObjectArray copy constructor");
                if constexpr (std::is_trivially_copyable<T>::value) 
                    hostDeep[i] = shallow; // For trivial types like double, just return the shallow copy.
                else
                    hostDeep[i] = shallow.clone(); // For non-trivial types, call clone() to perform a deep copy.
            }
            // Allocate new memory if necessary
            size_t sizeInBytes = static_cast<size_t>(other.m_size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy assignment");
            err = cudaMemcpy(m_data, hostDeep, sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy assignment: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }

            delete[] hostDeep; // Free the deep copy host array
#endif
        #endif
        } 
        else 
            m_data = nullptr;
    }

    // Move constructor
    __host__ __device__
    DeviceObjectArray(DeviceObjectArray&& other)
        : m_size(other.m_size), m_data(other.m_data)
    {
        other.m_size = 0;
        other.m_data = nullptr;
    }

    // Copy assignment operator
    __host__ __device__
    DeviceObjectArray& operator=(const DeviceObjectArray& other)
    {
        if (this != &other) 
        {
#if defined(__CUDA_ARCH__) //device branch
    #if 0
            // Free the old data
            //printf("DeviceObjectArray copy assignment operator used on device!\n");
            if (m_data) 
            {
                delete[] m_data;
                m_data = nullptr;
            }
            // Allocate new memory if necessary
            if(other.m_size > 0) 
            {
                m_data = new T[other.m_size];
                for (int i = 0; i < other.m_size; ++i) 
                    m_data[i] = other.m_data[i];
            }
            m_size = other.m_size;
            //printf("DeviceObjectArray copy assignment operator used on device!\n");
    #else
            // Destroy current elements by explicitly calling their destructors.
            for (size_t i = 0; i < m_size; ++i) 
                m_data[i].~T();
            // Free the raw memory if it was allocated.
            if (m_data) 
            {
                ::operator delete[](m_data);
                m_data = nullptr;
            }
            m_size = other.m_size;
            if (m_size > 0)
            {
                // Allocate raw memory without constructing objects.
                m_data = static_cast<T*>(::operator new[](m_size * sizeof(T)));
                if (m_data == nullptr)
                {
                    m_size = 0;
                    return *this;
                }
                for (size_t i = 0; i < m_size; ++i)
                    new (&m_data[i]) T(other.m_data[i]);
            }
    #endif
#else //end device branch; host branch start
            if (m_data) 
            {
                cudaFree(m_data);
                m_data = nullptr;
            }
            m_size = other.m_size;
    #if 1
        #if 0
            T* hostTemp = new T[m_size];
            for (int i = 0; i < m_size; ++i)
            {
                cudaError_t err = cudaMemcpy(&hostTemp[i],
                                             other.m_data + i,
                                             sizeof(T),
                                             cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess && "cudaMemcpy (device->host) failed in DeviceObjectArray copy constructor");
            }
            T* hostDeep = new T[m_size];
            for (int i = 0; i < m_size; ++i) 
            {
                if constexpr (std::is_trivially_copyable<T>::value) 
                    hostDeep[i] = hostTemp[i];
                else
                    hostDeep[i] = hostTemp[i].clone(); 
            }

            // Allocate new memory if necessary
            size_t sizeInBytes = static_cast<size_t>(other.m_size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy assignment");
            err = cudaMemcpy(m_data, hostDeep, sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy assignment: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }

            delete[] hostTemp; // Free the temporary host array
            delete[] hostDeep; // Free the deep copy host array
        #else
            #if 1
            size_t sizeInBytes = static_cast<size_t>(other.m_size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy assignment");
            //err = cudaMemcpy(m_data, other.m_data, sizeInBytes, cudaMemcpyHostToDevice);
            err = cudaMemcpy(m_data, other.m_data, sizeInBytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy assignment: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }
            #else
            size_t sizeInBytes = static_cast<size_t>(m_size * sizeof(T));
            T* h_Data = new T[m_size];
            cudaError_t err = cudaMemcpy(h_Data, other.m_data, sizeInBytes, cudaMemcpyDeviceToHost);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy constructor Device->Host");
            err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy constructor Host->Device");
            err = cudaMemcpy(m_data, h_Data, sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy constructor: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }
            delete [] h_Data;
            #endif
        #endif
    #else
            T* hostDeep = new T[m_size];
            for (int i = 0; i < m_size; ++i)
            {
                T shallow;
                cudaError_t err = cudaMemcpy(&shallow,
                                             other.m_data + i,
                                             sizeof(T),
                                             cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess && 
                "cudaMemcpy (device->host) failed in DeviceObjectArray copy constructor");
                if constexpr (std::is_trivially_copyable<T>::value) 
                    hostDeep[i] = shallow; // For trivial types like double, just return the shallow copy.
                else
                    hostDeep[i] = shallow.clone(); // For non-trivial types, call clone() to perform a deep copy.
            }
            // Allocate new memory if necessary
            size_t sizeInBytes = static_cast<size_t>(other.m_size * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray copy assignment");
            err = cudaMemcpy(m_data, hostDeep, sizeInBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray copy assignment: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }

            delete[] hostDeep; // Free the deep copy host array
    #endif 
            
#endif //end host branch
        }
        return *this;
    }

    // Move assignment operator
    __host__ __device__
    DeviceObjectArray& operator=(DeviceObjectArray&& other)
    {
        if (this != &other) 
        {
            // Free current memory
        #if defined(__CUDA_ARCH__)
            delete[] m_data;
        #else
            cudaFree(m_data);
        #endif
            // Steal other's data
            m_size = other.m_size;
            m_data = other.m_data;
            other.m_size = 0;
            other.m_data = nullptr;
        }
        return *this;
    }

    __host__ __device__
    ~DeviceObjectArray() 
    { 
    #if defined(__CUDA_ARCH__)
        #if 1
        if (!std::is_trivially_destructible<T>::value)
        {
            for (int i = 0; i < m_size; ++i) 
            {
                m_data[i].~T(); // Call destructor for each object
            }
        }
        else
        #endif
        delete[] m_data; 
    #else
        if (!std::is_trivially_destructible<T>::value)
        {
            if (m_size >0)
            {
                //printf("Launching destructKernel for %s.\n", typeid(T).name());
                int blockSize = 256;
                int numBlocks = (m_size + blockSize - 1) / blockSize;
                destructKernel<<<numBlocks, blockSize>>>(m_data, m_size);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) 
                    printf("Error in destructKernel<%s>: %s\n", typeid(T).name(), cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) 
                    printf("Error in destructKernel<%s>: %s\n", typeid(T).name(), cudaGetErrorString(err));
            }
            
        }
        if (m_data) 
        {
            cudaFree(m_data);
            m_data = nullptr;
        }
    #endif
    }

    __host__
    DeviceObjectArray<T> clone() const
    {
        DeviceObjectArray<T> copy(m_size);
        if (m_size > 0) 
        {
            size_t sizeInBytes = m_size * sizeof(T);
            // Allocate temporary host arrays
    #if 1
            T* hostTemp = new T[m_size];
            for (int i = 0; i < m_size; ++i)
            {
                // Copy data from device to host for deep copy
                cudaError_t err = cudaMemcpy(&hostTemp[i], m_data + i, sizeof(T), cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray clone");
            }
            T* hostDeep = new T[m_size];
            for (int i = 0; i < m_size; ++i) 
            {
                if constexpr (std::is_trivially_copyable<T>::value) 
                    hostDeep[i] = hostTemp[i];
                else
                    hostDeep[i] = hostTemp[i].clone(); 
            }
            cudaError_t err = cudaMalloc((void**)&(copy.m_data), sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in clone");
            // Copy the underlying data from this object
            err = cudaMemcpy(copy.m_data, hostDeep, sizeInBytes, cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in clone");
            delete[] hostTemp;
            delete[] hostDeep;
    #else
            T* hostDeep = new T[m_size];
            for (int i = 0; i < m_size; ++i)
            {
    #if 0
                hostDeep[i] = (*this)[i];
    #else
                cudaError_t err = cudaMemcpy(&hostDeep[i], m_data + i, sizeof(T), cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray clone");
    #endif
            }
            cudaError_t err = cudaMalloc((void**)&(copy.m_data), sizeInBytes);
            assert(err == cudaSuccess && "cudaMalloc failed in clone");
            // Copy the underlying data from this object
            err = cudaMemcpy(copy.m_data, hostDeep, sizeInBytes, cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in clone");
            delete[] hostDeep;
    #endif
           
        }
        return copy;
    }

#if 0
    template <typename T>
    typename std::enable_if<!std::is_trivially_copyable<T>::value, T>::type
    deepCopyReturn(const T* source) const
    {
        T temp;
        cudaMemcpy(&temp, source, sizeof(T), cudaMemcpyDeviceToHost);
        return temp.clone();
    }

    template <typename T>
    typename std::enable_if<std::is_trivially_copyable<T>::value, T>::type
    deepCopyReturn(const T* source) const
    {
        T temp;
        cudaMemcpy(&temp, source, sizeof(T), cudaMemcpyDeviceToHost);
        return temp;
    }
#endif

    __host__ __device__
    const T operator[](int index) const
    { 
    #if defined(__CUDA_ARCH__)
        return m_data[index];
    #else
#if 1
        T value;
        cudaError_t err =cudaMemcpy(&value, m_data + index, sizeof(T), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess && "cudaMemcpy failed in operator[]");
    #if 1
        T deepCopy = value;
        return value;
    #else
        if constexpr (std::is_trivially_copyable<T>::value)
            return value;
        else
            return value.clone();
    #endif
#else
        return deepCopyReturn<T>(m_data + index);
#endif
    #endif 
    }

    __host__ __device__
    void operator()(int size)
    {
    #if defined(__CUDA_ARCH__)
        delete[] m_data;
        m_size = size;
        m_data = new T[size];
    #else
        cudaFree(m_data);
        size_t sizeInBytes = static_cast<size_t>(size * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
        assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray operator()");
        m_size = size;
    #endif
    }

#if 0
    __device__
    const T& operator[](int index) const { return m_data[index]; }
#endif

    class Proxy
    {
    public:
        DeviceObjectArray* m_array;
        int m_index;

        __host__
        Proxy(DeviceObjectArray* array, int index) : m_array(array), m_index(index) {}

        // Const version of the constructor
        Proxy(const DeviceObjectArray* array, int index) : m_array(const_cast<DeviceObjectArray*>(array)), m_index(index) {}

        // Conversion operator for reading from device memory.
        __host__
        operator T() const
        {
#if 0
            T tmp;
            cudaError_t err = cudaMemcpy(&tmp, m_array->m_data + m_index, sizeof(T), cudaMemcpyDeviceToHost);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy conversion operator");
    #if 1
            if constexpr (std::is_trivially_copyable<T>::value) 
                // For trivial types like double, just return the shallow copy.
                return tmp;
            else 
                // For non-trivial types, call clone() to perform a deep copy.
                return tmp.clone();
    #else 
        #if 0
            const T deepCopy = tmp; // This will work for trivial types and non-trivial types as well.
            return deepCopy; 
        #else
            T deepCopy = tmp;
            return deepCopy;
        #endif
    #endif
#else
            T value;
            if constexpr (std::is_trivially_copyable<T>::value)
            {
                cudaError_t err = cudaMemcpy(&value, m_array->m_data + m_index, 
                                             sizeof(T), cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess && 
                "cudaMemcpy failed in DeviceObjectArray Proxy conversion operator");
            }
            else
            {
                T* d_value = nullptr;
                cudaError_t err = cudaMalloc((void**)&d_value, sizeof(T));
                assert(err == cudaSuccess && 
                "cudaMalloc failed in DeviceObjectArray Proxy conversion operator");
                DeviceObjectArray<T>* d_thisArray = nullptr;
                err = cudaMalloc((void**)&d_thisArray, sizeof(DeviceObjectArray<T>));
                assert(err == cudaSuccess &&
                "cudaMalloc failed in DeviceObjectArray Proxy conversion operator");
                err = cudaMemcpy(d_thisArray, this, sizeof(DeviceObjectArray<T>), cudaMemcpyHostToDevice);
                assert(err == cudaSuccess &&
                "cudaMemcpy failed in DeviceObjectArray Proxy conversion operator");
                deviceDeepCopyKernel<<<1, 1>>>(d_value, m_array->m_data + m_index);
                //retrieveDataKernel<<<1, 1>>>(d_value.data(),d_thisArray, m_index);
                err = cudaGetLastError();
                if (err != cudaSuccess) 
                    printf("Error in deviceDeepCopyKernel for %s: %s\n", typeid(T).name(), cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) 
                    printf("Error in deviceDeepCopyKernel for %s: %s\n", typeid(T).name(), cudaGetErrorString(err));
                err = cudaMemcpy(&value, d_value, sizeof(T), cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess &&
                "cudaMemcpy failed in DeviceObjectArray Proxy conversion operator");
                cudaFree(d_value);
                cudaFree(d_thisArray);
            }
            return value;

#endif
        }

        // Assignment operator for writing to device memory.
#if 0
        __host__
        T operator=(const T &value)
        {
#if 0
            const T deepCopy = value;
            cudaError_t err = cudaMemcpy(m_array->m_data + m_index, &deepCopy, sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy assignment operator");
            return deepCopy;
#else
            cudaError_t err = cudaMemcpy(m_array->m_data + m_index, &value, sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy assignment operator");
            return value;
#endif
        }

        __host__
        T operator=(T &&value)
        {
#if 1
#if 0
            const T deepCopy = value;
#else
            T deepCopy = value.clone();
#endif
            cudaError_t err = cudaMemcpy(m_array->m_data + m_index, &deepCopy, sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy assignment operator");
            return deepCopy;
#else
            cudaError_t err = cudaMemcpy(m_array->m_data + m_index, &value, sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy assignment operator");
            return value;
#endif
        }
#else
        __host__
        T operator=(const T &value)
        {
#if 0
            T deepCopy = value;
            cudaError_t err = cudaMemcpy(m_array->m_data + m_index, &deepCopy, sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy assignment operator");
            return deepCopy;
#else
            if constexpr (std::is_trivially_copyable<T>::value)
            {
                cudaError_t err = cudaMemcpy(m_array->m_data + m_index, &value, sizeof(T), cudaMemcpyHostToDevice);
                assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy assignment operator");
            }
            else
            {
                T* d_value = nullptr;
                cudaError_t err = cudaMalloc((void**)&d_value, sizeof(T));
                assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray Proxy assignment operator");
                err = cudaMemcpy(d_value, &value, sizeof(T), cudaMemcpyHostToDevice);
                assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray Proxy assignment operator");
                deviceDeepCopyKernel<<<1, 1>>>(m_array->m_data + m_index, d_value);
                err = cudaGetLastError();
                if (err != cudaSuccess) 
                    printf("Error in deviceDeepCopyKernel for %s: %s\n", typeid(T).name(), cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) 
                    printf("Error in deviceDeepCopyKernel for %s: %s\n", typeid(T).name(), cudaGetErrorString(err));
                cudaFree(d_value);
            }
            return value;
#endif
        }
#endif

    };
    
#if 0
    __host__ __device__
    #if defined(__CUDA_ARCH__) 
    T& operator[](int index) const
    {
        return m_data[index];
    }
    #else
    Proxy operator[](int index) const
    {
        return Proxy(this, index);
    }
    #endif // __CUDA_ARCH__

    __host__ __device__
    #if defined(__CUDA_ARCH__) 
    T& operator[](int index)
    { 
        return m_data[index]; 
    }
    #else
    Proxy operator[](int index)
    {
        return Proxy(this, index);
    }
    #endif // __CUDA_ARCH__
#else
    __device__
    T& operator[](int index)
    { 
        return m_data[index]; 
    }
#endif

    __host__
    Proxy at(int index) const
    {
        return Proxy(this, index);
    }

    __host__
    Proxy at(int index)
    {
        return Proxy(this, index);
    }

    __host__ __device__ 
    int size() const 
    { 
    //#if defined(__CUDA_ARCH__)
        return m_size; 
    //#else
    //    int h_size;
    //    cudaMemcpy(&h_size, &m_size, sizeof(int), cudaMemcpyDeviceToHost);
    //    return h_size;
    //#endif
    }

    __host__ __device__
    void resize(int newSize)
    {
    #if defined(__CUDA_ARCH__)
        delete[] m_data;
        m_size = newSize;
        m_data = new T[newSize];
    #else
        cudaFree(m_data);
        size_t sizeInBytes = static_cast<size_t>(newSize * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
        assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray resize");
        m_size = newSize;
    #endif
    }

    __host__ __device__
    void clear()
    {
    #if defined(__CUDA_ARCH__)
        delete[] m_data;
        m_size = 0;
        m_data = nullptr;
    #else
        if (!std::is_trivially_destructible<T>::value)
        {
            if (m_size >0)
            {
                int blockSize = 256;
                int numBlocks = (m_size + blockSize - 1) / blockSize;
                destructKernel<<<numBlocks, blockSize>>>(m_data, m_size);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) 
                    printf("Error in destructKernel<%s>: %s\n", typeid(T).name(), cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) 
                    printf("Error in destructKernel<%s>: %s\n", typeid(T).name(), cudaGetErrorString(err));
            }
            
        }
        if (m_data) 
        {
            cudaFree(m_data);
            m_data = nullptr;
        }
        m_size = 0;
    #endif
    }

    __host__ __device__
    T* data()
    { 
        return m_data;
    }

    __host__
    std::vector<T> retrieveDataToHost() const
    {
        if (m_size > 0) 
        {
            std::vector<T> hostData(m_size); // Create a vector to hold the data on the host
            T* hostData_ptr = new T[m_size]; // Allocate a temporary host array to hold the data
            cudaError_t err = cudaMemcpy(hostData_ptr, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) 
            {
                printf("cudaMemcpy failed in DeviceObjectArray retrieveDataToHost: %s\n", cudaGetErrorString(err));
                assert(err == cudaSuccess);
            }
            // Copy the data from the temporary host array to the vector
            for (int i = 0; i < m_size; ++i) 
            {
                hostData[i] = hostData_ptr[i];
            }
            delete[] hostData_ptr; // Free the temporary host array
            return hostData; // Return the vector containing the data on the host
        }
        else 
        {
            return std::vector<T>(); // Return an empty vector if size is 0
        }
    }

    __host__ __device__
    const T* data() const { return m_data; }

    __host__ __device__
    void push_back(const T& value)
    {
    #if defined(__CUDA_ARCH__)
        T* newData = new T[m_size + 1];
        for (int i = 0; i < m_size; ++i)
            newData[i] = m_data[i];
        newData[m_size] = value;
        delete[] m_data;
        m_data = newData;
        ++m_size;
    #else
        T* newData = nullptr;
        cudaMalloc((void**)&newData, (m_size + 1) * sizeof(T));
        cudaMemcpy(newData, m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(newData + m_size, &value, sizeof(T), cudaMemcpyHostToDevice);
        cudaFree(m_data);
        m_data = newData;
        ++m_size;
    #endif
    }

    __host__ __device__
    void setData(const T* data, int size)
    {
    #if defined(__CUDA_ARCH__)
        delete[] m_data;
        m_size = size;
        m_data = new T[size];
        for (int i = 0; i < size; ++i)
            m_data[i] = data[i];
    #else
        cudaFree(m_data);
        size_t sizeInBytes = static_cast<size_t>(size * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, sizeInBytes);
        assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectArray setData");
        err = cudaMemcpy(m_data, data, sizeInBytes, cudaMemcpyHostToDevice);
        assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray setData");
        m_size = size;
    #endif
    }

    __host__
    void parallelDataSetting(const T* values, int size)
    {
        assert(m_size == size && "Size mismatch in parallelDataSetting");
        size_t sizeInBytes = static_cast<size_t>(size * sizeof(T));
        T* d_values;
        cudaMalloc((void**)&d_values, sizeInBytes);
        cudaMemcpy(d_values, values, sizeInBytes, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        deviceAssign<<<numBlocks, blockSize>>>(m_data, size, d_values);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            fprintf(stderr, "Error in deviceAssign: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
            fprintf(stderr, "Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaFree(d_values);
    }

    __host__
    void parallelDataReading(T* values, int size)
    {
        assert(m_size == size && "Size mismatch in parallelDataReading");
        size_t sizeInBytes = static_cast<size_t>(size * sizeof(T));
        T* d_values;
        cudaMalloc((void**)&d_values, sizeInBytes);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        deviceRead<<<numBlocks, blockSize>>>(d_values, size, m_data);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            fprintf(stderr, "Error in deviceRead: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
            fprintf(stderr, "Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaMemcpy(values, d_values, sizeInBytes, cudaMemcpyDeviceToHost);
        cudaFree(d_values);
    }

    __host__ __device__
    void print(/*const char* chs = ""*/) const
    {
        //printf("%s", chs);
        //printf("111:");
    #if defined(__CUDA_ARCH__)
        for (int i = 0; i < m_size; ++i)
            printElement(m_data[i]);
    #else
        T* hostData = new T[m_size];
        cudaMemcpy(hostData, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost);
        for (int i = 0; i < m_size; ++i)
            printElement(hostData[i]);
        delete[] hostData;
    #endif
        printf("\n");
    }

    //probmatic, only works for trivial types or types with a proper printElement function
    __host__ __device__
    void swap(DeviceObjectArray& other)
    {
        std::swap(m_size, other.m_size);
        std::swap(m_data, other.m_data);
    }

    __host__ __device__
    const T back() const
    {
    #if defined(__CUDA_ARCH__)
        return m_data[m_size - 1];
    #else
        T value;
        cudaMemcpy(&value, m_data + m_size - 1, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    #endif
    }

    __host__
    void shallowCopyToHost(T* hostData) const
    {
        if (m_size > 0) 
        {
            size_t sizeInBytes = m_size * sizeof(T);
            cudaError_t err = cudaMemcpy(hostData, m_data, sizeInBytes, cudaMemcpyDeviceToHost);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectArray shallowCopyToHost");
        }
    }
};


template <typename T>
__global__
void deviceAssign(T* data, int size, const T* value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) 
        data[idx] = value[idx];
}

template <typename T>
__global__
void deviceRead(T* value, int size, const T* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) 
    {
        printf("deviceRead: %d\n", idx);
        value[idx] = data[idx];
    }
}

template <typename T>
__global__ 
void deviceDeepCopyKernel(T* device_dst, T* host_src) 
{
    //printf("deviceDeepCopyKernel called\n");
    new (device_dst) T(*host_src);
    //device_dst->operator=(*host_src); // Call the assignment operator to copy the object
    //printf("deviceDeepCopyKernel done\n");
}

template <typename T>
__global__ 
void destructKernel(T* ptr, size_t count) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) 
    {
        //printf("Destructing object at index %d.\n", idx);
        ptr[idx].~T();
        //printf("Destructed object at index %d.\n", idx);
    }
}

#if 0
template <typename T>
__global__
void retrieveDataKernel(T* data, DeviceObjectArray<T>* d_thisArray, 
                        int index, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) 
    {
        data[idx] = (*d_thisArray)[index].data()[idx];
    }
}
#endif